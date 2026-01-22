import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
class URWIDRepl(repl.Repl):
    _time_between_redraws = 0.05

    def __init__(self, event_loop, palette, interpreter, config):
        super().__init__(interpreter, config)
        self._redraw_handle = None
        self._redraw_pending = False
        self._redraw_time = 0
        self.listbox = BPythonListBox(urwid.SimpleListWalker([]))
        self.tooltip = urwid.ListBox(urwid.SimpleListWalker([]))
        self.tooltip.grid = None
        self.overlay = Tooltip(self.listbox, self.tooltip)
        self.stdout_hist = ''
        self.frame = urwid.Frame(self.overlay)
        if urwid.get_encoding_mode() == 'narrow':
            input_filter = decoding_input_filter
        else:
            input_filter = None
        self.main_loop = urwid.MainLoop(self.frame, palette, event_loop=event_loop, unhandled_input=self.handle_input, input_filter=input_filter, handle_mouse=False)
        self.statusbar = Statusbar(config, _(' <%s> Rewind  <%s> Save  <%s> Pastebin  <%s> Pager  <%s> Show Source ') % (config.undo_key, config.save_key, config.pastebin_key, config.last_output_key, config.show_source_key), self.main_loop)
        self.frame.set_footer(self.statusbar.widget)
        self.interact = URWIDInteraction(self.config, self.statusbar, self.frame)
        self.edits = []
        self.edit = None
        self.current_output = None
        self._completion_update_suppressed = False
        self.exit_value = ()
        load_urwid_command_map(config)

    def echo(self, orig_s):
        s = orig_s.rstrip('\n')
        if s:
            if self.current_output is None:
                self.current_output = urwid.Text(('output', s))
                if self.edit is None:
                    self.listbox.body.append(self.current_output)
                    self.listbox.set_focus(len(self.listbox.body) - 1)
                else:
                    self.listbox.body.insert(-1, self.current_output)
                    self.listbox.set_focus(len(self.listbox.body) - 1)
            else:
                self.current_output.set_text(('output', self.current_output.text + s))
        if orig_s.endswith('\n'):
            self.current_output = None
        if self._redraw_handle is None:
            self.main_loop.draw_screen()

            def maybe_redraw(loop, self):
                if self._redraw_pending:
                    loop.draw_screen()
                    self._redraw_pending = False
                self._redraw_handle = None
            self._redraw_handle = self.main_loop.set_alarm_in(self._time_between_redraws, maybe_redraw, self)
            self._redraw_time = time.time()
        else:
            self._redraw_pending = True
            now = time.time()
            if now - self._redraw_time > 2 * self._time_between_redraws:
                self.main_loop.draw_screen()
                self._redraw_time = now

    def _get_current_line(self):
        if self.edit is None:
            return ''
        return self.edit.get_edit_text()

    def _set_current_line(self, line):
        self.edit.set_edit_text(line)
    current_line = property(_get_current_line, _set_current_line, None, 'Return the current line (the one the cursor is in).')

    def cw(self):
        """Return the current word (incomplete word left of cursor)."""
        if self.edit is None:
            return
        pos = self.edit.edit_pos
        text = self.edit.get_edit_text()
        if pos != len(text):
            return
        if not text or (not text[-1].isalnum() and text[-1] not in ('.', '_')):
            return
        for i, c in enumerate(reversed(text)):
            if not c.isalnum() and c not in ('.', '_'):
                break
        else:
            return text
        return text[-i:]

    @property
    def cpos(self):
        if self.edit is not None:
            return len(self.current_line) - self.edit.edit_pos
        return 0

    def _get_cursor_offset(self):
        return self.edit.edit_pos

    def _set_cursor_offset(self, offset):
        self.edit.edit_pos = offset
    cursor_offset = property(_get_cursor_offset, _set_cursor_offset, None, 'The cursor offset from the beginning of the line')

    def _populate_completion(self):
        widget_list = self.tooltip.body
        while widget_list:
            widget_list.pop()
        if self.complete():
            if self.funcprops:
                func_name = self.funcprops.func
                args = self.funcprops.argspec.args
                is_bound = self.funcprops.is_bound_method
                in_arg = self.arg_pos
                varargs = self.funcprops.argspec.varargs
                varkw = self.funcprops.argspec.varkwargs
                defaults = self.funcprops.argspec.defaults
                kwonly = self.funcprops.argspec.kwonly
                kwonly_defaults = self.funcprops.argspec.kwonly_defaults or {}
                markup = [('bold name', func_name), ('name', ': (')]
                if is_bound and isinstance(in_arg, int):
                    in_arg += 1
                for k, i in enumerate(args):
                    if defaults and k + 1 > len(args) - len(defaults):
                        kw = repr(defaults[k - (len(args) - len(defaults))])
                    else:
                        kw = None
                    if not k and str(i) == 'self':
                        color = 'name'
                    else:
                        color = 'token'
                    if k == in_arg or i == in_arg:
                        color = 'bold ' + color
                    markup.append((color, str(i)))
                    if kw is not None:
                        markup.extend([('punctuation', '='), ('token', kw)])
                    if k != len(args) - 1:
                        markup.append(('punctuation', ', '))
                if varargs:
                    if args:
                        markup.append(('punctuation', ', '))
                    markup.append(('token', '*' + varargs))
                if kwonly:
                    if not varargs:
                        if args:
                            markup.append(('punctuation', ', '))
                        markup.append(('punctuation', '*'))
                    for arg in kwonly:
                        if arg == in_arg:
                            color = 'bold token'
                        else:
                            color = 'token'
                        markup.extend([('punctuation', ', '), (color, arg)])
                        if arg in kwonly_defaults:
                            markup.extend([('punctuation', '='), ('token', repr(kwonly_defaults[arg]))])
                if varkw:
                    if args or varargs or kwonly:
                        markup.append(('punctuation', ', '))
                    markup.append(('token', '**' + varkw))
                markup.append(('punctuation', ')'))
                widget_list.append(urwid.Text(markup))
            if self.matches_iter.matches:
                attr_map = {}
                focus_map = {'main': 'operator'}
                texts = [urwid.AttrMap(urwid.Text(('main', match)), attr_map, focus_map) for match in self.matches_iter.matches]
                width = max((text.original_widget.pack()[0] for text in texts))
                gridflow = urwid.GridFlow(texts, width, 1, 0, 'left')
                widget_list.append(gridflow)
                self.tooltip.grid = gridflow
                self.overlay.tooltip_focus = False
            else:
                self.tooltip.grid = None
            self.frame.body = self.overlay
        else:
            self.frame.body = self.listbox
            self.tooltip.grid = None
        if self.docstring:
            docstring = self.docstring
            widget_list.append(urwid.Text(('comment', docstring)))

    def reprint_line(self, lineno, tokens):
        edit = self.edits[-len(self.buffer) + lineno - 1]
        edit.set_edit_markup(list(format_tokens(tokens)))

    def getstdout(self):
        """This method returns the 'spoofed' stdout buffer, for writing to a
        file or sending to a pastebin or whatever."""
        return self.stdout_hist + '\n'

    def ask_confirmation(self, q):
        """Ask for yes or no and return boolean"""
        try:
            reply = self.statusbar.prompt(q)
        except ValueError:
            return False
        return reply.lower() in ('y', 'yes')

    def reevaluate(self):
        """Clear the buffer, redraw the screen and re-evaluate the history"""
        self.evaluating = True
        self.stdout_hist = ''
        self.f_string = ''
        self.buffer = []
        self.scr.erase()
        self.cpos = -1
        self.prompt(False)
        self.iy, self.ix = self.scr.getyx()
        for line in self.history:
            self.stdout_hist += line + '\n'
            self.print_line(line)
            self.scr.addstr('\n')
            more = self.push(line)
            self.prompt(more)
            self.iy, self.ix = self.scr.getyx()
        self.cpos = 0
        indent = repl.next_indentation(self.s, self.config.tab_length)
        self.s = ''
        self.scr.refresh()
        if self.buffer:
            for unused in range(indent):
                self.tab()
        self.evaluating = False

    def write(self, s):
        """For overriding stdout defaults"""
        if '\x04' in s:
            for block in s.split('\x04'):
                self.write(block)
            return
        if s.rstrip() and '\x03' in s:
            t = s.split('\x03')[1]
        else:
            t = s
        if not self.stdout_hist:
            self.stdout_hist = t
        else:
            self.stdout_hist += t
        self.echo(s)

    def push(self, s, insert_into_history=True):
        orig_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            return super().push(s, insert_into_history)
        except SystemExit as e:
            self.exit_value = e.args
            raise urwid.ExitMainLoop()
        except KeyboardInterrupt:
            self.keyboard_interrupt()
        finally:
            signal.signal(signal.SIGINT, orig_handler)

    def start(self):
        self.prompt(False)

    def keyboard_interrupt(self):
        if self.edit is not None:
            self.edit.make_readonly()
            self.edit = None
            self.buffer = []
            self.echo('KeyboardInterrupt')
            self.prompt(False)
        else:
            self.echo('KeyboardInterrupt')

    def prompt(self, more):
        self.current_output = None
        self.rl_history.reset()
        if not more:
            caption = ('prompt', self.ps1)
            self.stdout_hist += self.ps1
        else:
            caption = ('prompt_more', self.ps2)
            self.stdout_hist += self.ps2
        self.edit = BPythonEdit(self.config, caption=caption)
        urwid.connect_signal(self.edit, 'change', self.on_input_change)
        urwid.connect_signal(self.edit, 'edit-pos-changed', self.on_edit_pos_changed)
        self.edit.insert_text(4 * self.next_indentation() * ' ')
        self.edits.append(self.edit)
        self.listbox.body.append(self.edit)
        self.listbox.set_focus(len(self.listbox.body) - 1)
        self.frame.body = self.listbox

    def on_input_change(self, edit, text):
        tokens = self.tokenize(text, False)
        edit.set_edit_markup(list(format_tokens(tokens)))
        if not self._completion_update_suppressed:
            self.main_loop.set_alarm_in(0, lambda *args: self._populate_completion())

    def on_edit_pos_changed(self, edit, position):
        """Gets called when the cursor position inside the edit changed.
        Rehighlight the current line because there might be a paren under
        the cursor now."""
        tokens = self.tokenize(self.current_line, False)
        edit.set_edit_markup(list(format_tokens(tokens)))

    def handle_input(self, event):
        if self.frame.get_focus() != 'body':
            return
        if event == 'enter':
            inp = self.edit.get_edit_text()
            self.history.append(inp)
            self.edit.make_readonly()
            self.stdout_hist += inp
            self.stdout_hist += '\n'
            self.edit = None
            self.main_loop.draw_screen()
            more = self.push(inp)
            self.prompt(more)
        elif event == 'ctrl d':
            if self.edit is not None:
                if not self.edit.get_edit_text():
                    raise urwid.ExitMainLoop()
                else:
                    self.main_loop.process_input(['delete'])
        elif urwid.command_map[event] == 'cursor up':
            self.rl_history.enter(self.edit.get_edit_text())
            self.edit.set_edit_text('')
            self.edit.insert_text(self.rl_history.back())
        elif urwid.command_map[event] == 'cursor down':
            self.rl_history.enter(self.edit.get_edit_text())
            self.edit.set_edit_text('')
            self.edit.insert_text(self.rl_history.forward())
        elif urwid.command_map[event] == 'next selectable':
            self.tab()
        elif urwid.command_map[event] == 'prev selectable':
            self.tab(True)

    def tab(self, back=False):
        """Process the tab key being hit.

        If the line is blank or has only whitespace: indent.

        If there is text before the cursor: cycle completions.

        If `back` is True cycle backwards through completions, and return
        instead of indenting.

        Returns True if the key was handled.
        """
        self._completion_update_suppressed = True
        try:
            text = self.edit.get_edit_text()
            if not text.lstrip() and (not back):
                x_pos = len(text) - self.cpos
                num_spaces = x_pos % self.config.tab_length
                if not num_spaces:
                    num_spaces = self.config.tab_length
                self.edit.insert_text(' ' * num_spaces)
                return True
            if not self.matches_iter:
                self.complete(tab=True)
                cw = self.current_string() or self.cw()
                if not cw:
                    return True
            if self.matches_iter.is_cseq():
                cursor, text = self.matches_iter.substitute_cseq()
                self.edit.set_edit_text(text)
                self.edit.edit_pos = cursor
            elif self.matches_iter.matches:
                if back:
                    self.matches_iter.previous()
                else:
                    next(self.matches_iter)
                cursor, text = self.matches_iter.cur_line()
                self.edit.set_edit_text(text)
                self.edit.edit_pos = cursor
                self.overlay.tooltip_focus = True
                if self.tooltip.grid:
                    self.tooltip.grid.set_focus(self.matches_iter.index)
            return True
        finally:
            self._completion_update_suppressed = False