from __future__ import unicode_literals
import functools
import os
import signal
import six
import sys
import textwrap
import threading
import time
import types
import weakref
from subprocess import Popen
from .application import Application, AbortAction
from .buffer import Buffer
from .buffer_mapping import BufferMapping
from .completion import CompleteEvent, get_common_complete_suffix
from .enums import SEARCH_BUFFER
from .eventloop.base import EventLoop
from .eventloop.callbacks import EventLoopCallbacks
from .filters import Condition
from .input import StdinInput, Input
from .key_binding.input_processor import InputProcessor
from .key_binding.input_processor import KeyPress
from .key_binding.registry import Registry
from .key_binding.vi_state import ViState
from .keys import Keys
from .output import Output
from .renderer import Renderer, print_tokens
from .search_state import SearchState
from .utils import Event
from .buffer import AcceptAction
class CommandLineInterface(object):
    """
    Wrapper around all the other classes, tying everything together.

    Typical usage::

        application = Application(...)
        cli = CommandLineInterface(application, eventloop)
        result = cli.run()
        print(result)

    :param application: :class:`~prompt_toolkit.application.Application` instance.
    :param eventloop: The :class:`~prompt_toolkit.eventloop.base.EventLoop` to
                      be used when `run` is called. The easiest way to create
                      an eventloop is by calling
                      :meth:`~prompt_toolkit.shortcuts.create_eventloop`.
    :param input: :class:`~prompt_toolkit.input.Input` instance.
    :param output: :class:`~prompt_toolkit.output.Output` instance. (Probably
                   Vt100_Output or Win32Output.)
    """

    def __init__(self, application, eventloop=None, input=None, output=None):
        assert isinstance(application, Application)
        assert isinstance(eventloop, EventLoop), 'Passing an eventloop is required.'
        assert output is None or isinstance(output, Output)
        assert input is None or isinstance(input, Input)
        from .shortcuts import create_output
        self.application = application
        self.eventloop = eventloop
        self._is_running = False
        self.output = output or create_output()
        self.input = input or StdinInput(sys.stdin)
        assert isinstance(application.buffers, BufferMapping)
        self.buffers = application.buffers
        self.editing_mode = application.editing_mode
        self.quoted_insert = False
        self.vi_state = ViState()
        self.renderer = Renderer(self.application.style, self.output, use_alternate_screen=application.use_alternate_screen, mouse_support=application.mouse_support)
        self.render_counter = 0
        self.max_render_postpone_time = 0
        self._invalidated = False
        self.input_processor = InputProcessor(application.key_bindings_registry, weakref.ref(self))
        self._async_completers = {}
        self._sub_cli = None
        for name, b in self.buffers.items():
            self.add_buffer(name, b)
        self.on_buffer_changed = Event(self, application.on_buffer_changed)
        self.on_initialize = Event(self, application.on_initialize)
        self.on_input_timeout = Event(self, application.on_input_timeout)
        self.on_invalidate = Event(self, application.on_invalidate)
        self.on_render = Event(self, application.on_render)
        self.on_reset = Event(self, application.on_reset)
        self.on_start = Event(self, application.on_start)
        self.on_stop = Event(self, application.on_stop)
        self.reset()
        self.on_initialize += self.application.on_initialize
        self.on_initialize.fire()

    @property
    def layout(self):
        return self.application.layout

    @property
    def clipboard(self):
        return self.application.clipboard

    @property
    def pre_run_callables(self):
        return self.application.pre_run_callables

    def add_buffer(self, name, buffer, focus=False):
        """
        Insert a new buffer.
        """
        assert isinstance(buffer, Buffer)
        self.buffers[name] = buffer
        if focus:
            self.buffers.focus(name)
        auto_suggest_function = self._create_auto_suggest_function(buffer)
        completer_function = self._create_async_completer(buffer)
        self._async_completers[name] = completer_function

        def create_on_insert_handler():
            """
            Wrapper around the asynchronous completer and auto suggestion, that
            ensures that it's only called while typing if the
            `complete_while_typing` filter is enabled.
            """

            def on_text_insert(_):
                if buffer.completer and buffer.complete_while_typing():
                    completer_function()
                if buffer.auto_suggest:
                    auto_suggest_function()
            return on_text_insert
        buffer.on_text_insert += create_on_insert_handler()

        def buffer_changed(_):
            """
            When the text in a buffer changes.
            (A paste event is also a change, but not an insert. So we don't
            want to do autocompletions in this case, but we want to propagate
            the on_buffer_changed event.)
            """
            self.on_buffer_changed.fire()
        buffer.on_text_changed += buffer_changed

    def start_completion(self, buffer_name=None, select_first=False, select_last=False, insert_common_part=False, complete_event=None):
        """
        Start asynchronous autocompletion of this buffer.
        (This will do nothing if a previous completion was still in progress.)
        """
        buffer_name = buffer_name or self.current_buffer_name
        completer = self._async_completers.get(buffer_name)
        if completer:
            completer(select_first=select_first, select_last=select_last, insert_common_part=insert_common_part, complete_event=CompleteEvent(completion_requested=True))

    @property
    def current_buffer_name(self):
        """
        The name of the current  :class:`.Buffer`. (Or `None`.)
        """
        return self.buffers.current_name(self)

    @property
    def current_buffer(self):
        """
        The currently focussed :class:`~.Buffer`.

        (This returns a dummy :class:`.Buffer` when none of the actual buffers
        has the focus. In this case, it's really not practical to check for
        `None` values or catch exceptions every time.)
        """
        return self.buffers.current(self)

    def focus(self, buffer_name):
        """
        Focus the buffer with the given name on the focus stack.
        """
        self.buffers.focus(self, buffer_name)

    def push_focus(self, buffer_name):
        """
        Push to the focus stack.
        """
        self.buffers.push_focus(self, buffer_name)

    def pop_focus(self):
        """
        Pop from the focus stack.
        """
        self.buffers.pop_focus(self)

    @property
    def terminal_title(self):
        """
        Return the current title to be displayed in the terminal.
        When this in `None`, the terminal title remains the original.
        """
        result = self.application.get_title()
        assert result is None or isinstance(result, six.text_type)
        return result

    @property
    def is_searching(self):
        """
        True when we are searching.
        """
        return self.current_buffer_name == SEARCH_BUFFER

    def reset(self, reset_current_buffer=False):
        """
        Reset everything, for reading the next input.

        :param reset_current_buffer: XXX: not used anymore. The reason for
            having this option in the past was when this CommandLineInterface
            is run multiple times, that we could reset the buffer content from
            the previous run. This is now handled in the AcceptAction.
        """
        self._exit_flag = False
        self._abort_flag = False
        self._return_value = None
        self.renderer.reset()
        self.input_processor.reset()
        self.layout.reset()
        self.vi_state.reset()
        self.search_state = SearchState(ignore_case=Condition(lambda: self.is_ignoring_case))
        self.on_reset.fire()

    @property
    def in_paste_mode(self):
        """ True when we are in paste mode. """
        return self.application.paste_mode(self)

    @property
    def is_ignoring_case(self):
        """ True when we currently ignore casing. """
        return self.application.ignore_case(self)

    def invalidate(self):
        """
        Thread safe way of sending a repaint trigger to the input event loop.
        """
        if self._invalidated:
            return
        else:
            self._invalidated = True
        self.on_invalidate.fire()
        if self.eventloop is not None:

            def redraw():
                self._invalidated = False
                self._redraw()
            if self.max_render_postpone_time:
                _max_postpone_until = time.time() + self.max_render_postpone_time
            else:
                _max_postpone_until = None
            self.eventloop.call_from_executor(redraw, _max_postpone_until=_max_postpone_until)
    request_redraw = invalidate

    def _redraw(self):
        """
        Render the command line again. (Not thread safe!) (From other threads,
        or if unsure, use :meth:`.CommandLineInterface.invalidate`.)
        """
        if self._is_running and self._sub_cli is None:
            self.render_counter += 1
            self.renderer.render(self, self.layout, is_done=self.is_done)
            self.on_render.fire()

    def _on_resize(self):
        """
        When the window size changes, we erase the current output and request
        again the cursor position. When the CPR answer arrives, the output is
        drawn again.
        """
        self.renderer.erase(leave_alternate_screen=False, erase_title=False)
        self.renderer.request_absolute_cursor_position()
        self._redraw()

    def _load_next_buffer_indexes(self):
        for buff, index in self._next_buffer_indexes.items():
            if buff in self.buffers:
                self.buffers[buff].working_index = index

    def _pre_run(self, pre_run=None):
        """ Called during `run`. """
        if pre_run:
            pre_run()
        for c in self.pre_run_callables:
            c()
        del self.pre_run_callables[:]

    def run(self, reset_current_buffer=False, pre_run=None):
        """
        Read input from the command line.
        This runs the eventloop until a return value has been set.

        :param reset_current_buffer: XXX: Not used anymore.
        :param pre_run: Callable that is called right after the reset has taken
            place. This allows custom initialisation.
        """
        assert pre_run is None or callable(pre_run)
        try:
            self._is_running = True
            self.on_start.fire()
            self.reset()
            self._pre_run(pre_run)
            with self.input.raw_mode():
                self.renderer.request_absolute_cursor_position()
                self._redraw()
                self.eventloop.run(self.input, self.create_eventloop_callbacks())
        finally:
            if not self.is_done:
                self._exit_flag = True
                self._redraw()
            self.renderer.reset()
            self.on_stop.fire()
            self._is_running = False
        return self.return_value()
    try:
        six.exec_(textwrap.dedent('\n        def run_async(self, reset_current_buffer=True, pre_run=None):\n            """\n            Same as `run`, but this returns a coroutine.\n\n            This is only available on Python >3.3, with asyncio.\n            """\n            # Inline import, because it slows down startup when asyncio is not\n            # needed.\n            import asyncio\n\n            @asyncio.coroutine\n            def run():\n                assert pre_run is None or callable(pre_run)\n\n                try:\n                    self._is_running = True\n\n                    self.on_start.fire()\n                    self.reset()\n\n                    # Call pre_run.\n                    self._pre_run(pre_run)\n\n                    with self.input.raw_mode():\n                        self.renderer.request_absolute_cursor_position()\n                        self._redraw()\n\n                        yield from self.eventloop.run_as_coroutine(\n                                self.input, self.create_eventloop_callbacks())\n\n                    return self.return_value()\n                finally:\n                    if not self.is_done:\n                        self._exit_flag = True\n                        self._redraw()\n\n                    self.renderer.reset()\n                    self.on_stop.fire()\n                    self._is_running = False\n\n            return run()\n        '))
    except SyntaxError:

        def run_async(self, reset_current_buffer=True, pre_run=None):
            """
            Same as `run`, but this returns a coroutine.

            This is only available on Python >3.3, with asyncio.
            """
            raise NotImplementedError

    def run_sub_application(self, application, done_callback=None, erase_when_done=False, _from_application_generator=False):
        """
        Run a sub :class:`~prompt_toolkit.application.Application`.

        This will suspend the main application and display the sub application
        until that one returns a value. The value is returned by calling
        `done_callback` with the result.

        The sub application will share the same I/O of the main application.
        That means, it uses the same input and output channels and it shares
        the same event loop.

        .. note:: Technically, it gets another Eventloop instance, but that is
            only a proxy to our main event loop. The reason is that calling
            'stop' --which returns the result of an application when it's
            done-- is handled differently.
        """
        assert isinstance(application, Application)
        assert done_callback is None or callable(done_callback)
        if self._sub_cli is not None:
            raise RuntimeError('Another sub application started already.')
        if not _from_application_generator:
            self.renderer.erase()

        def done():
            sub_cli._redraw()
            if erase_when_done or application.erase_when_done:
                sub_cli.renderer.erase()
            sub_cli.renderer.reset()
            sub_cli._is_running = False
            self._sub_cli = None
            if not _from_application_generator:
                self.renderer.request_absolute_cursor_position()
                self._redraw()
            if done_callback:
                done_callback(sub_cli.return_value())
        sub_cli = CommandLineInterface(application=application, eventloop=_SubApplicationEventLoop(self, done), input=self.input, output=self.output)
        sub_cli._is_running = True
        sub_cli._redraw()
        self._sub_cli = sub_cli

    def exit(self):
        """
        Set exit. When Control-D has been pressed.
        """
        on_exit = self.application.on_exit
        self._exit_flag = True
        self._redraw()
        if on_exit == AbortAction.RAISE_EXCEPTION:

            def eof_error():
                raise EOFError()
            self._set_return_callable(eof_error)
        elif on_exit == AbortAction.RETRY:
            self.reset()
            self.renderer.request_absolute_cursor_position()
            self.current_buffer.reset()
        elif on_exit == AbortAction.RETURN_NONE:
            self.set_return_value(None)

    def abort(self):
        """
        Set abort. When Control-C has been pressed.
        """
        on_abort = self.application.on_abort
        self._abort_flag = True
        self._redraw()
        if on_abort == AbortAction.RAISE_EXCEPTION:

            def keyboard_interrupt():
                raise KeyboardInterrupt()
            self._set_return_callable(keyboard_interrupt)
        elif on_abort == AbortAction.RETRY:
            self.reset()
            self.renderer.request_absolute_cursor_position()
            self.current_buffer.reset()
        elif on_abort == AbortAction.RETURN_NONE:
            self.set_return_value(None)
    set_exit = exit
    set_abort = abort

    def set_return_value(self, document):
        """
        Set a return value. The eventloop can retrieve the result it by calling
        `return_value`.
        """
        self._set_return_callable(lambda: document)
        self._redraw()

    def _set_return_callable(self, value):
        assert callable(value)
        self._return_value = value
        if self.eventloop:
            self.eventloop.stop()

    def run_in_terminal(self, func, render_cli_done=False):
        """
        Run function on the terminal above the prompt.

        What this does is first hiding the prompt, then running this callable
        (which can safely output to the terminal), and then again rendering the
        prompt which causes the output of this function to scroll above the
        prompt.

        :param func: The callable to execute.
        :param render_cli_done: When True, render the interface in the
                'Done' state first, then execute the function. If False,
                erase the interface first.

        :returns: the result of `func`.
        """
        if render_cli_done:
            self._return_value = True
            self._redraw()
            self.renderer.reset()
        else:
            self.renderer.erase()
        self._return_value = None
        with self.input.cooked_mode():
            result = func()
        self.renderer.reset()
        self.renderer.request_absolute_cursor_position()
        self._redraw()
        return result

    def run_application_generator(self, coroutine, render_cli_done=False):
        """
        EXPERIMENTAL
        Like `run_in_terminal`, but takes a generator that can yield Application instances.

        Example:

            def f():
                yield Application1(...)
                print('...')
                yield Application2(...)
            cli.run_in_terminal_async(f)

        The values which are yielded by the given coroutine are supposed to be
        `Application` instances that run in the current CLI, all other code is
        supposed to be CPU bound, so except for yielding the applications,
        there should not be any user interaction or I/O in the given function.
        """
        if render_cli_done:
            self._return_value = True
            self._redraw()
            self.renderer.reset()
        else:
            self.renderer.erase()
        self._return_value = None
        g = coroutine()
        assert isinstance(g, types.GeneratorType)

        def step_next(send_value=None):
            """ Execute next step of the coroutine."""
            try:
                with self.input.cooked_mode():
                    result = g.send(send_value)
            except StopIteration:
                done()
            except:
                done()
                raise
            else:
                assert isinstance(result, Application)
                self.run_sub_application(result, done_callback=step_next, _from_application_generator=True)

        def done():
            self.renderer.reset()
            self.renderer.request_absolute_cursor_position()
            self._redraw()
        step_next()

    def run_system_command(self, command):
        """
        Run system command (While hiding the prompt. When finished, all the
        output will scroll above the prompt.)

        :param command: Shell command to be executed.
        """

        def wait_for_enter():
            """
            Create a sub application to wait for the enter key press.
            This has two advantages over using 'input'/'raw_input':
            - This will share the same input/output I/O.
            - This doesn't block the event loop.
            """
            from .shortcuts import create_prompt_application
            registry = Registry()

            @registry.add_binding(Keys.ControlJ)
            @registry.add_binding(Keys.ControlM)
            def _(event):
                event.cli.set_return_value(None)
            application = create_prompt_application(message='Press ENTER to continue...', key_bindings_registry=registry)
            self.run_sub_application(application)

        def run():
            try:
                input_fd = self.input.fileno()
            except AttributeError:
                input_fd = sys.stdin.fileno()
            try:
                output_fd = self.output.fileno()
            except AttributeError:
                output_fd = sys.stdout.fileno()
            p = Popen(command, shell=True, stdin=input_fd, stdout=output_fd)
            p.wait()
            wait_for_enter()
        self.run_in_terminal(run)

    def suspend_to_background(self, suspend_group=True):
        """
        (Not thread safe -- to be called from inside the key bindings.)
        Suspend process.

        :param suspend_group: When true, suspend the whole process group.
            (This is the default, and probably what you want.)
        """
        if hasattr(signal, 'SIGTSTP'):

            def run():
                if suspend_group:
                    os.kill(0, signal.SIGTSTP)
                else:
                    os.kill(os.getpid(), signal.SIGTSTP)
            self.run_in_terminal(run)

    def print_tokens(self, tokens, style=None):
        """
        Print a list of (Token, text) tuples to the output.
        (When the UI is running, this method has to be called through
        `run_in_terminal`, otherwise it will destroy the UI.)

        :param style: Style class to use. Defaults to the active style in the CLI.
        """
        print_tokens(self.output, tokens, style or self.application.style)

    @property
    def is_exiting(self):
        """
        ``True`` when the exit flag as been set.
        """
        return self._exit_flag

    @property
    def is_aborting(self):
        """
        ``True`` when the abort flag as been set.
        """
        return self._abort_flag

    @property
    def is_returning(self):
        """
        ``True`` when a return value has been set.
        """
        return self._return_value is not None

    def return_value(self):
        """
        Get the return value. Not that this method can throw an exception.
        """
        if self._return_value:
            return self._return_value()

    @property
    def is_done(self):
        return self.is_exiting or self.is_aborting or self.is_returning

    def _create_async_completer(self, buffer):
        """
        Create function for asynchronous autocompletion.
        (Autocomplete in other thread.)
        """
        complete_thread_running = [False]

        def completion_does_nothing(document, completion):
            """
            Return `True` if applying this completion doesn't have any effect.
            (When it doesn't insert any new text.
            """
            text_before_cursor = document.text_before_cursor
            replaced_text = text_before_cursor[len(text_before_cursor) + completion.start_position:]
            return replaced_text == completion.text

        def async_completer(select_first=False, select_last=False, insert_common_part=False, complete_event=None):
            document = buffer.document
            complete_event = complete_event or CompleteEvent(text_inserted=True)
            if complete_thread_running[0]:
                return
            if buffer.complete_state or not buffer.completer:
                return
            complete_thread_running[0] = True

            def run():
                completions = list(buffer.completer.get_completions(document, complete_event))

                def callback():
                    """
                    Set the new complete_state in a safe way. Don't replace an
                    existing complete_state if we had one. (The user could have
                    pressed 'Tab' in the meantime. Also don't set it if the text
                    was changed in the meantime.
                    """
                    complete_thread_running[0] = False
                    if len(completions) == 1 and completion_does_nothing(document, completions[0]):
                        del completions[:]
                    if buffer.text == document.text and buffer.cursor_position == document.cursor_position and (not buffer.complete_state):
                        set_completions = True
                        select_first_anyway = False
                        if insert_common_part:
                            common_part = get_common_complete_suffix(document, completions)
                            if common_part:
                                buffer.insert_text(common_part)
                                if len(completions) > 1:
                                    completions[:] = [c.new_completion_from_position(len(common_part)) for c in completions]
                                else:
                                    set_completions = False
                            elif len(completions) == 1:
                                select_first_anyway = True
                        if set_completions:
                            buffer.set_completions(completions=completions, go_to_first=select_first or select_first_anyway, go_to_last=select_last)
                        self.invalidate()
                    elif not buffer.complete_state:
                        async_completer()
                if self.eventloop:
                    self.eventloop.call_from_executor(callback)
            self.eventloop.run_in_executor(run)
        return async_completer

    def _create_auto_suggest_function(self, buffer):
        """
        Create function for asynchronous auto suggestion.
        (AutoSuggest in other thread.)
        """
        suggest_thread_running = [False]

        def async_suggestor():
            document = buffer.document
            if suggest_thread_running[0]:
                return
            if buffer.suggestion or not buffer.auto_suggest:
                return
            suggest_thread_running[0] = True

            def run():
                suggestion = buffer.auto_suggest.get_suggestion(self, buffer, document)

                def callback():
                    suggest_thread_running[0] = False
                    if buffer.text == document.text and buffer.cursor_position == document.cursor_position:
                        buffer.suggestion = suggestion
                        self.invalidate()
                    else:
                        async_suggestor()
                if self.eventloop:
                    self.eventloop.call_from_executor(callback)
            self.eventloop.run_in_executor(run)
        return async_suggestor

    def stdout_proxy(self, raw=False):
        """
        Create an :class:`_StdoutProxy` class which can be used as a patch for
        `sys.stdout`. Writing to this proxy will make sure that the text
        appears above the prompt, and that it doesn't destroy the output from
        the renderer.

        :param raw: (`bool`) When True, vt100 terminal escape sequences are not
                    removed/escaped.
        """
        return _StdoutProxy(self, raw=raw)

    def patch_stdout_context(self, raw=False, patch_stdout=True, patch_stderr=True):
        """
        Return a context manager that will replace ``sys.stdout`` with a proxy
        that makes sure that all printed text will appear above the prompt, and
        that it doesn't destroy the output from the renderer.

        :param patch_stdout: Replace `sys.stdout`.
        :param patch_stderr: Replace `sys.stderr`.
        """
        return _PatchStdoutContext(self.stdout_proxy(raw=raw), patch_stdout=patch_stdout, patch_stderr=patch_stderr)

    def create_eventloop_callbacks(self):
        return _InterfaceEventLoopCallbacks(self)