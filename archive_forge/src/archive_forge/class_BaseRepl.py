import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
class BaseRepl(Repl):
    """Python Repl

    Reacts to events like
     - terminal dimensions and change events
     - keystrokes
    Behavior altered by
     - number of scroll downs that were necessary to render array after each
       display
     - initial cursor position
    outputs:
     - 2D array to be rendered

    BaseRepl is mostly view-independent state of Repl - but self.width and
    self.height are important for figuring out how to wrap lines for example.
    Usually self.width and self.height should be set by receiving a window
    resize event, not manually set to anything - as long as the first event
    received is a window resize event, this works fine.

    Subclasses are responsible for implementing several methods.
    """

    def __init__(self, config: Config, window: CursorAwareWindow, locals_: Optional[Dict[str, Any]]=None, banner: Optional[str]=None, interp: Optional[Interp]=None, orig_tcattrs: Optional[List[Any]]=None):
        """
        locals_ is a mapping of locals to pass into the interpreter
        config is a bpython config.Struct with config attributes
        banner is a string to display briefly in the status bar
        interp is an interpreter instance to use
        original terminal state, useful for shelling out with normal terminal
        """
        logger.debug('starting init')
        self.window = window
        self.weak_rewind = bool(locals_ or interp)
        if interp is None:
            interp = Interp(locals=locals_)
            interp.write = self.send_to_stdouterr
        if banner is None:
            if config.help_key:
                banner = _('Welcome to bpython!') + ' ' + _('Press <%s> for help.') % config.help_key
            else:
                banner = None
        if config.cli_suggestion_width <= 0 or config.cli_suggestion_width > 1:
            config.cli_suggestion_width = 1
        self.reevaluating = False
        self.fake_refresh_requested = False
        self.status_bar = StatusBar(config, '', request_refresh=self.request_refresh, schedule_refresh=self.schedule_refresh)
        self.edit_keys = edit_keys.mapping_with_config(config, key_dispatch)
        logger.debug('starting parent init')
        super().__init__(interp, config)
        self.formatter = BPythonFormatter(config.color_scheme)
        self.interact = self.status_bar
        self._current_line = ''
        self.current_stdouterr_line: Union[str, FmtStr] = ''
        self.display_lines: List[FmtStr] = []
        self.history = []
        self.all_logical_lines: List[Tuple[str, LineType]] = []
        self.display_buffer: List[FmtStr] = []
        self.scroll_offset = 0
        self._cursor_offset = 0
        self.orig_tcattrs: Optional[List[Any]] = orig_tcattrs
        self.coderunner = CodeRunner(self.interp, self.request_refresh)
        self.stdout = FakeOutput(self.coderunner, self.send_to_stdouterr, real_fileobj=sys.__stdout__)
        self.stderr = FakeOutput(self.coderunner, self.send_to_stdouterr, real_fileobj=sys.__stderr__)
        self.stdin = FakeStdin(self.coderunner, self, self.edit_keys)
        self.request_paint_to_clear_screen = False
        self.request_paint_to_pad_bottom = 0
        self.inconsistent_history = False
        self.history_already_messed_up = False
        self.last_events: List[Optional[str]] = [None] * 50
        self.presentation_mode = False
        self.paste_mode = False
        self.current_match = None
        self.list_win_visible = False
        self.watching_files = config.default_autoreload
        self.incr_search_mode = SearchMode.NO_SEARCH
        self.incr_search_target = ''
        self.original_modules = set(sys.modules.keys())
        try:
            self.width, self.height = os.get_terminal_size()
        except OSError:
            self.width = -1
            self.height = -1
        self.status_bar.message(banner)
        self.watcher = ModuleChangedEventHandler([], self.request_reload)
        if self.watcher and config.default_autoreload:
            self.watcher.activate()

    def get_cursor_vertical_diff(self):
        """Return how the cursor moved due to a window size change"""
        return 0

    def get_top_usable_line(self):
        """Return the top line of display that can be rewritten"""
        return 0

    def get_term_hw(self):
        """Returns the current width and height of the display area."""
        return (50, 10)

    def _schedule_refresh(self, when: float):
        """Arrange for the bpython display to be refreshed soon.

        This method will be called when the Repl wants the display to be
        refreshed at a known point in the future, and as such it should
        interrupt a pending request to the user for input.

        Because the worst-case effect of not refreshing
        is only having an out of date UI until the user enters input, a
        default NOP implementation is provided."""

    def _request_refresh(self):
        """Arrange for the bpython display to be refreshed soon.

        This method will be called when the Repl wants to refresh the display,
        but wants control returned to it afterwards. (it is assumed that simply
        returning from process_event will cause an event refresh)

        The very next event received by process_event should be a
        RefreshRequestEvent."""
        raise NotImplementedError

    def _request_reload(self, files_modified: Sequence[str]) -> None:
        """Like request_refresh, but for reload requests events."""
        raise NotImplementedError

    def request_undo(self, n=1):
        """Like request_refresh, but for undo request events."""
        raise NotImplementedError

    def on_suspend(self):
        """Will be called on sigtstp.

        Do whatever cleanup would allow the user to use other programs."""
        raise NotImplementedError

    def after_suspend(self):
        """Will be called when process foregrounded after suspend.

        See to it that process_event is called with None to trigger a refresh
        if not in the middle of a process_event call when suspend happened."""
        raise NotImplementedError

    def request_refresh(self):
        """Request that the bpython display to be refreshed soon."""
        if self.reevaluating or self.paste_mode:
            self.fake_refresh_requested = True
        else:
            self._request_refresh()

    def request_reload(self, files_modified: Sequence[str]=()) -> None:
        """Request that a ReloadEvent be passed next into process_event"""
        if self.watching_files:
            self._request_reload(files_modified)

    def schedule_refresh(self, when: float=0) -> None:
        """Schedule a ScheduledRefreshRequestEvent for when.

        Such a event should interrupt if blocked waiting for keyboard input"""
        if self.reevaluating or self.paste_mode:
            self.fake_refresh_requested = True
        else:
            self._schedule_refresh(when=when)

    def __enter__(self):
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.orig_stdin = sys.stdin
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        sys.stdin = self.stdin
        self.orig_sigwinch_handler = signal.getsignal(signal.SIGWINCH)
        self.orig_sigtstp_handler = signal.getsignal(signal.SIGTSTP)
        if is_main_thread():
            signal.signal(signal.SIGWINCH, self.sigwinch_handler)
            signal.signal(signal.SIGTSTP, self.sigtstp_handler)
        self.orig_meta_path = sys.meta_path
        if self.watcher:
            meta_path = []
            for finder in sys.meta_path:
                meta_path.append(ImportFinder(self.watcher, finder))
            sys.meta_path = meta_path
        sitefix.monkeypatch_quit()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        sys.stdin = self.orig_stdin
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        if is_main_thread():
            signal.signal(signal.SIGWINCH, self.orig_sigwinch_handler)
            signal.signal(signal.SIGTSTP, self.orig_sigtstp_handler)
        sys.meta_path = self.orig_meta_path
        return False

    def sigwinch_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        old_rows, old_columns = (self.height, self.width)
        self.height, self.width = self.get_term_hw()
        cursor_dy = self.get_cursor_vertical_diff()
        self.scroll_offset -= cursor_dy
        logger.info('sigwinch! Changed from %r to %r', (old_rows, old_columns), (self.height, self.width))
        logger.info('decreasing scroll offset by %d to %d', cursor_dy, self.scroll_offset)

    def sigtstp_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        self.scroll_offset = len(self.lines_for_display)
        self.__exit__(None, None, None)
        self.on_suspend()
        os.kill(os.getpid(), signal.SIGTSTP)
        self.after_suspend()
        self.__enter__()

    def clean_up_current_line_for_exit(self):
        """Called when trying to exit to prep for final paint"""
        logger.debug('unhighlighting paren for exit')
        self.cursor_offset = -1
        self.unhighlight_paren()

    def process_event(self, e: Union[events.Event, str]) -> Optional[bool]:
        """Returns True if shutting down, otherwise returns None.
        Mostly mutates state of Repl object"""
        logger.debug('processing event %r', e)
        if isinstance(e, events.Event):
            return self.process_control_event(e)
        else:
            self.last_events.append(e)
            self.last_events.pop(0)
            self.process_key_event(e)
            return None

    def process_control_event(self, e: events.Event) -> Optional[bool]:
        if isinstance(e, bpythonevents.ScheduledRefreshRequestEvent):
            pass
        elif isinstance(e, bpythonevents.RefreshRequestEvent):
            logger.info('received ASAP refresh request event')
            if self.status_bar.has_focus:
                self.status_bar.process_event(e)
            else:
                assert self.coderunner.code_is_waiting
                self.run_code_and_maybe_finish()
        elif self.status_bar.has_focus:
            self.status_bar.process_event(e)
        elif isinstance(e, events.PasteEvent):
            ctrl_char = compress_paste_event(e)
            if ctrl_char is not None:
                return self.process_event(ctrl_char)
            with self.in_paste_mode():
                if len(e.events) <= MAX_EVENTS_POSSIBLY_NOT_PASTE and any((not is_simple_event(ee) for ee in e.events)):
                    for ee in e.events:
                        if self.stdin.has_focus:
                            self.stdin.process_event(ee)
                        else:
                            self.process_event(ee)
                else:
                    simple_events = just_simple_events(e.events)
                    source = preprocess(''.join(simple_events), self.interp.compile)
                    for ee in source:
                        if self.stdin.has_focus:
                            self.stdin.process_event(ee)
                        else:
                            self.process_simple_keypress(ee)
        elif isinstance(e, bpythonevents.RunStartupFileEvent):
            try:
                self.startup()
            except OSError as err:
                self.status_bar.message(_('Executing PYTHONSTARTUP failed: %s') % (err,))
        elif isinstance(e, bpythonevents.UndoEvent):
            self.undo(n=e.n)
        elif self.stdin.has_focus:
            self.stdin.process_event(e)
        elif isinstance(e, events.SigIntEvent):
            logger.debug('received sigint event')
            self.keyboard_interrupt()
        elif isinstance(e, bpythonevents.ReloadEvent):
            if self.watching_files:
                self.clear_modules_and_reevaluate()
                self.status_bar.message(_('Reloaded at %s because %s modified.') % (time.strftime('%X'), ' & '.join(e.files_modified)))
        else:
            raise ValueError("Don't know how to handle event type: %r" % e)
        return None

    def process_key_event(self, e: str) -> None:
        if self.status_bar.has_focus:
            return self.status_bar.process_event(e)
        if self.stdin.has_focus:
            return self.stdin.process_event(e)
        if e in key_dispatch[self.config.right_key] + key_dispatch[self.config.end_of_line_key] + ('<RIGHT>',) and self.config.curtsies_right_arrow_completion and (self.cursor_offset == len(self.current_line)):
            self.current_line += self.current_suggestion
            self.cursor_offset = len(self.current_line)
        elif e in ('<UP>',) + key_dispatch[self.config.up_one_line_key]:
            self.up_one_line()
        elif e in ('<DOWN>',) + key_dispatch[self.config.down_one_line_key]:
            self.down_one_line()
        elif e == '<Ctrl-d>':
            self.on_control_d()
        elif e == '<Ctrl-o>':
            self.operate_and_get_next()
        elif e == '<Esc+.>':
            self.get_last_word()
        elif e in key_dispatch[self.config.reverse_incremental_search_key]:
            self.incremental_search(reverse=True)
        elif e in key_dispatch[self.config.incremental_search_key]:
            self.incremental_search()
        elif e in ('<BACKSPACE>',) + key_dispatch[self.config.backspace_key] and self.incr_search_mode != SearchMode.NO_SEARCH:
            self.add_to_incremental_search(self, backspace=True)
        elif e in self.edit_keys.cut_buffer_edits:
            self.readline_kill(e)
        elif e in self.edit_keys.simple_edits:
            self.cursor_offset, self.current_line = self.edit_keys.call(e, cursor_offset=self.cursor_offset, line=self.current_line, cut_buffer=self.cut_buffer)
        elif e in key_dispatch[self.config.cut_to_buffer_key]:
            self.cut_to_buffer()
        elif e in key_dispatch[self.config.reimport_key]:
            self.clear_modules_and_reevaluate()
        elif e in key_dispatch[self.config.toggle_file_watch_key]:
            self.toggle_file_watch()
        elif e in key_dispatch[self.config.clear_screen_key]:
            self.request_paint_to_clear_screen = True
        elif e in key_dispatch[self.config.show_source_key]:
            self.show_source()
        elif e in key_dispatch[self.config.help_key]:
            self.pager(self.help_text())
        elif e in key_dispatch[self.config.exit_key]:
            raise SystemExit()
        elif e in ('\n', '\r', '<PADENTER>', '<Ctrl-j>', '<Ctrl-m>'):
            self.on_enter()
        elif e == '<TAB>':
            self.on_tab()
        elif e == '<Shift-TAB>':
            self.on_tab(back=True)
        elif e in key_dispatch[self.config.undo_key]:
            self.prompt_undo()
        elif e in key_dispatch[self.config.redo_key]:
            self.redo()
        elif e in key_dispatch[self.config.save_key]:
            greenlet.greenlet(self.write2file).switch()
        elif e in key_dispatch[self.config.pastebin_key]:
            greenlet.greenlet(self.pastebin).switch()
        elif e in key_dispatch[self.config.copy_clipboard_key]:
            greenlet.greenlet(self.copy2clipboard).switch()
        elif e in key_dispatch[self.config.external_editor_key]:
            self.send_session_to_external_editor()
        elif e in key_dispatch[self.config.edit_config_key]:
            greenlet.greenlet(self.edit_config).switch()
        elif e in key_dispatch[self.config.edit_current_block_key]:
            self.send_current_block_to_external_editor()
        elif e == '<ESC>':
            self.incr_search_mode = SearchMode.NO_SEARCH
        elif e == '<SPACE>':
            self.add_normal_character(' ')
        elif e in CHARACTER_PAIR_MAP.keys():
            if e in ["'", '"']:
                if self.is_closing_quote(e):
                    self.insert_char_pair_end(e)
                else:
                    self.insert_char_pair_start(e)
            else:
                self.insert_char_pair_start(e)
        elif e in CHARACTER_PAIR_MAP.values():
            self.insert_char_pair_end(e)
        else:
            self.add_normal_character(e)

    def is_closing_quote(self, e: str) -> bool:
        char_count = self._current_line.count(e)
        return char_count % 2 == 0 and cursor_on_closing_char_pair(self._cursor_offset, self._current_line, e)[0]

    def insert_char_pair_start(self, e):
        """Accepts character which is a part of CHARACTER_PAIR_MAP
        like brackets and quotes, and appends it to the line with
        an appropriate character pair ending. Closing character can only be inserted
        when the next character is either a closing character or a space

        e.x. if you type "(" (lparen) , this will insert "()"
        into the line
        """
        self.add_normal_character(e)
        if self.config.brackets_completion:
            start_of_line = len(self._current_line) == 1
            end_of_line = len(self._current_line) == self._cursor_offset
            can_lookup_next = len(self._current_line) > self._cursor_offset
            next_char = None if not can_lookup_next else self._current_line[self._cursor_offset]
            if start_of_line or end_of_line or (next_char is not None and next_char in '})] '):
                self.add_normal_character(CHARACTER_PAIR_MAP[e], narrow_search=False)
                self._cursor_offset -= 1

    def insert_char_pair_end(self, e):
        """Accepts character which is a part of CHARACTER_PAIR_MAP
        like brackets and quotes, and checks whether it should be
        inserted to the line or overwritten

        e.x. if you type ")" (rparen) , and your cursor is directly
        above another ")" (rparen) in the cmd, this will just skip
        it and move the cursor.
        If there is no same character underneath the cursor, the
        character will be printed/appended to the line
        """
        if self.config.brackets_completion:
            if self.cursor_offset < len(self._current_line):
                if self._current_line[self.cursor_offset] == e:
                    self.cursor_offset += 1
                    return
        self.add_normal_character(e)

    def get_last_word(self):
        previous_word = _last_word(self.rl_history.entry)
        word = _last_word(self.rl_history.back())
        line = self.current_line
        self._set_current_line(line[:len(line) - len(previous_word)] + word, reset_rl_history=False)
        self._set_cursor_offset(self.cursor_offset - len(previous_word) + len(word), reset_rl_history=False)

    def incremental_search(self, reverse=False, include_current=False):
        if self.incr_search_mode == SearchMode.NO_SEARCH:
            self.rl_history.enter(self.current_line)
            self.incr_search_target = ''
        elif self.incr_search_target:
            line = self.rl_history.back(False, search=True, target=self.incr_search_target, include_current=include_current) if reverse else self.rl_history.forward(False, search=True, target=self.incr_search_target, include_current=include_current)
            self._set_current_line(line, reset_rl_history=False, clear_special_mode=False)
            self._set_cursor_offset(len(self.current_line), reset_rl_history=False, clear_special_mode=False)
        if reverse:
            self.incr_search_mode = SearchMode.REVERSE_INCREMENTAL_SEARCH
        else:
            self.incr_search_mode = SearchMode.INCREMENTAL_SEARCH

    def readline_kill(self, e):
        func = self.edit_keys[e]
        self.cursor_offset, self.current_line, cut = func(self.cursor_offset, self.current_line)
        if self.last_events[-2] == e:
            if func.kills == 'ahead':
                self.cut_buffer += cut
            elif func.kills == 'behind':
                self.cut_buffer = cut + self.cut_buffer
            else:
                raise ValueError("cut value other than 'ahead' or 'behind'")
        else:
            self.cut_buffer = cut

    def on_enter(self, new_code=True, reset_rl_history=True):
        if new_code:
            self.redo_stack = []
        self._set_cursor_offset(-1, update_completion=False)
        if reset_rl_history:
            self.rl_history.reset()
        self.history.append(self.current_line)
        self.all_logical_lines.append((self.current_line, LineType.INPUT))
        self.push(self.current_line, insert_into_history=new_code)

    def on_tab(self, back=False):
        """Do something on tab key
        taken from bpython.cli

        Does one of the following:
        1) add space to move up to the next %4==0 column
        2) complete the current word with characters common to all completions
        3) select the first or last match
        4) select the next or previous match if already have a match
        """

        def only_whitespace_left_of_cursor():
            """returns true if all characters before cursor are whitespace"""
            return not self.current_line[:self.cursor_offset].strip()
        logger.debug('self.matches_iter.matches: %r', self.matches_iter.matches)
        if only_whitespace_left_of_cursor():
            front_ws = len(self.current_line[:self.cursor_offset]) - len(self.current_line[:self.cursor_offset].lstrip())
            to_add = 4 - front_ws % self.config.tab_length
            for unused in range(to_add):
                self.add_normal_character(' ')
            return
        if self.config.brackets_completion:
            on_closing_char, _ = cursor_on_closing_char_pair(self._cursor_offset, self._current_line)
            if on_closing_char:
                self._cursor_offset += 1
        if len(self.matches_iter.matches) == 0:
            self.list_win_visible = self.complete(tab=True)
        if self.matches_iter.is_cseq():
            cursor_and_line = self.matches_iter.substitute_cseq()
            self._cursor_offset, self._current_line = cursor_and_line
            if not self.matches_iter.matches:
                self.list_win_visible = self.complete()
        elif self.matches_iter.matches:
            self.current_match = back and self.matches_iter.previous() or next(self.matches_iter)
            cursor_and_line = self.matches_iter.cur_line()
            self._cursor_offset, self._current_line = cursor_and_line
            self.list_win_visible = True
        if self.config.brackets_completion:
            if self.is_completion_callable(self._current_line):
                self._current_line = self.append_closing_character(self._current_line)

    def is_completion_callable(self, completion):
        """Checks whether given completion is callable (e.x. function)"""
        completion_end = completion[-1]
        return completion_end in CHARACTER_PAIR_MAP

    def append_closing_character(self, completion):
        """Appends closing character/bracket to the completion"""
        completion_end = completion[-1]
        if completion_end in CHARACTER_PAIR_MAP:
            completion = f'{completion}{CHARACTER_PAIR_MAP[completion_end]}'
        return completion

    def on_control_d(self):
        if self.current_line == '':
            raise SystemExit()
        else:
            self.current_line = self.current_line[:self.cursor_offset] + self.current_line[self.cursor_offset + 1:]

    def cut_to_buffer(self):
        self.cut_buffer = self.current_line[self.cursor_offset:]
        self.current_line = self.current_line[:self.cursor_offset]

    def yank_from_buffer(self):
        pass

    def operate_and_get_next(self):
        self.on_enter(reset_rl_history=False)

    def up_one_line(self):
        self.rl_history.enter(self.current_line)
        self._set_current_line(tabs_to_spaces(self.rl_history.back(False, search=self.config.curtsies_right_arrow_completion)), update_completion=False, reset_rl_history=False)
        self._set_cursor_offset(len(self.current_line), reset_rl_history=False)

    def down_one_line(self):
        self.rl_history.enter(self.current_line)
        self._set_current_line(tabs_to_spaces(self.rl_history.forward(False, search=self.config.curtsies_right_arrow_completion)), update_completion=False, reset_rl_history=False)
        self._set_cursor_offset(len(self.current_line), reset_rl_history=False)

    def process_simple_keypress(self, e: str):
        if e in ('<Ctrl-j>', '<Ctrl-m>', '<PADENTER>', '\n', '\r'):
            self.on_enter()
            while self.fake_refresh_requested:
                self.fake_refresh_requested = False
                self.process_event(bpythonevents.RefreshRequestEvent())
        elif isinstance(e, events.Event):
            pass
        elif e == '<SPACE>':
            self.add_normal_character(' ')
        else:
            self.add_normal_character(e)

    def send_current_block_to_external_editor(self, filename=None):
        """
        Sends the current code block to external editor to be edited. Usually bound to C-x.
        """
        text = self.send_to_external_editor(self.get_current_block())
        lines = [line for line in text.split('\n')]
        while lines and (not lines[-1].split()):
            lines.pop()
        events = '\n'.join(lines + ([''] if len(lines) == 1 else ['', '']))
        self.clear_current_block()
        with self.in_paste_mode():
            for e in events:
                self.process_simple_keypress(e)
        self.cursor_offset = len(self.current_line)

    def send_session_to_external_editor(self, filename=None):
        """
        Sends entire bpython session to external editor to be edited. Usually bound to F7.
        """
        for_editor = EDIT_SESSION_HEADER
        for_editor += self.get_session_formatted_for_file()
        text = self.send_to_external_editor(for_editor)
        if text == for_editor:
            self.status_bar.message(_('Session not reevaluated because it was not edited'))
            return
        lines = text.split('\n')
        if len(lines) and (not lines[-1].strip()):
            lines.pop()
        if len(lines) and lines[-1].startswith('### '):
            current_line = lines[-1][4:]
        else:
            current_line = ''
        from_editor = [line for line in lines if line[:6] != '# OUT:' and line[:3] != '###']
        if all((not line.strip() for line in from_editor)):
            self.status_bar.message(_('Session not reevaluated because saved file was blank'))
            return
        source = preprocess('\n'.join(from_editor), self.interp.compile)
        lines = source.split('\n')
        self.history = lines
        self.reevaluate(new_code=True)
        self.current_line = current_line
        self.cursor_offset = len(self.current_line)
        self.status_bar.message(_('Session edited and reevaluated'))

    def clear_modules_and_reevaluate(self):
        if self.watcher:
            self.watcher.reset()
        cursor, line = (self.cursor_offset, self.current_line)
        for modname in set(sys.modules.keys()) - self.original_modules:
            del sys.modules[modname]
        self.reevaluate(new_code=False)
        self.cursor_offset, self.current_line = (cursor, line)
        self.status_bar.message(_('Reloaded at %s by user.') % (time.strftime('%X'),))

    def toggle_file_watch(self):
        if self.watcher:
            if self.watching_files:
                msg = _('Auto-reloading deactivated.')
                self.status_bar.message(msg)
                self.watcher.deactivate()
                self.watching_files = False
            else:
                msg = _('Auto-reloading active, watching for file changes...')
                self.status_bar.message(msg)
                self.watching_files = True
                self.watcher.activate()
        else:
            self.status_bar.message(_('Auto-reloading not available because watchdog not installed.'))

    def add_normal_character(self, char, narrow_search=True):
        if len(char) > 1 or is_nop(char):
            return
        if self.incr_search_mode != SearchMode.NO_SEARCH:
            self.add_to_incremental_search(char)
        else:
            self._set_current_line(self.current_line[:self.cursor_offset] + char + self.current_line[self.cursor_offset:], update_completion=False, reset_rl_history=False, clear_special_mode=False)
            if narrow_search:
                self.cursor_offset += 1
            else:
                self._cursor_offset += 1
        if self.config.cli_trim_prompts and self.current_line.startswith(self.ps1):
            self.current_line = self.current_line[4:]
            if narrow_search:
                self.cursor_offset = max(0, self.cursor_offset - 4)
            else:
                self._cursor_offset += max(0, self.cursor_offset - 4)

    def add_to_incremental_search(self, char=None, backspace=False):
        """Modify the current search term while in incremental search.

        The only operations allowed in incremental search mode are
        adding characters and backspacing."""
        if backspace:
            self.incr_search_target = self.incr_search_target[:-1]
        elif char is not None:
            self.incr_search_target += char
        else:
            raise ValueError('must provide a char or set backspace to True')
        if self.incr_search_mode == SearchMode.REVERSE_INCREMENTAL_SEARCH:
            self.incremental_search(reverse=True, include_current=True)
        elif self.incr_search_mode == SearchMode.INCREMENTAL_SEARCH:
            self.incremental_search(include_current=True)
        else:
            raise ValueError('add_to_incremental_search not in a special mode')

    def update_completion(self, tab=False):
        """Update visible docstring and matches and box visibility"""
        self.current_match = None
        self.list_win_visible = self.complete(tab)

    def predicted_indent(self, line):
        logger.debug('line is %r', line)
        indent = len(re.match('[ ]*', line).group())
        if line.endswith(':'):
            indent = max(0, indent + self.config.tab_length)
        elif line and line.count(' ') == len(line):
            indent = max(0, indent - self.config.tab_length)
        elif line and ':' not in line and line.strip().startswith(('return', 'pass', '...', 'raise', 'yield', 'break', 'continue')):
            indent = max(0, indent - self.config.tab_length)
        logger.debug('indent we found was %s', indent)
        return indent

    def push(self, line, insert_into_history=True):
        """Push a line of code onto the buffer, start running the buffer

        If the interpreter successfully runs the code, clear the buffer
        """
        if self.paste_mode:
            self.saved_indent = 0
        else:
            self.saved_indent = self.predicted_indent(line)
        if self.config.syntax:
            display_line = bpythonparse(pygformat(self.tokenize(line), self.formatter))
            logger.debug('display line being pushed to buffer: %r -> %r', line, display_line)
            self.display_buffer.append(display_line)
        else:
            self.display_buffer.append(fmtstr(line))
        if insert_into_history:
            self.insert_into_history(line)
        self.buffer.append(line)
        code_to_run = '\n'.join(self.buffer)
        logger.debug('running %r in interpreter', self.buffer)
        c, code_will_parse = code_finished_will_parse('\n'.join(self.buffer), self.interp.compile)
        self.saved_predicted_parse_error = not code_will_parse
        if c:
            logger.debug('finished - buffer cleared')
            self.cursor_offset = 0
            self.display_lines.extend(self.display_buffer_lines)
            self.display_buffer = []
            self.buffer = []
        self.coderunner.load_code(code_to_run)
        self.run_code_and_maybe_finish()

    def run_code_and_maybe_finish(self, for_code=None):
        r = self.coderunner.run_code(for_code=for_code)
        if r:
            logger.debug('----- Running finish command stuff -----')
            logger.debug('saved_indent: %r', self.saved_indent)
            err = self.saved_predicted_parse_error
            self.saved_predicted_parse_error = False
            indent = self.saved_indent
            if err:
                indent = 0
            if self.rl_history.index == 0:
                self._set_current_line(' ' * indent, update_completion=True)
            else:
                self._set_current_line(self.rl_history.entries[-self.rl_history.index], reset_rl_history=False)
            self.cursor_offset = len(self.current_line)

    def keyboard_interrupt(self):
        self.cursor_offset = -1
        self.unhighlight_paren()
        self.display_lines.extend(self.display_buffer_lines)
        self.display_lines.extend(paint.display_linize(self.current_cursor_line, self.width))
        self.display_lines.extend(paint.display_linize('KeyboardInterrupt', self.width))
        self.clear_current_block(remove_from_history=False)

    def unhighlight_paren(self):
        """Modify line in self.display_buffer to unhighlight a paren if
        possible.

        self.highlighted_paren should be a line in ?
        """
        if self.highlighted_paren is not None and self.config.syntax:
            lineno, saved_tokens = self.highlighted_paren
            if lineno == len(self.display_buffer):
                return
            self.highlighted_paren = None
            logger.debug('trying to unhighlight a paren on line %r', lineno)
            logger.debug('with these tokens: %r', saved_tokens)
            new = bpythonparse(pygformat(saved_tokens, self.formatter))
            self.display_buffer[lineno] = self.display_buffer[lineno].setslice_with_length(0, len(new), new, len(self.display_buffer[lineno]))

    def clear_current_block(self, remove_from_history=True):
        self.display_buffer = []
        if remove_from_history:
            del self.history[-len(self.buffer):]
            del self.all_logical_lines[-len(self.buffer):]
        self.buffer = []
        self.cursor_offset = 0
        self.saved_indent = 0
        self.current_line = ''
        self.cursor_offset = len(self.current_line)

    def get_current_block(self):
        """
        Returns the current code block as string (without prompts)
        """
        return '\n'.join(self.buffer + [self.current_line])

    def send_to_stdouterr(self, output):
        """Send unicode strings or FmtStr to Repl stdout or stderr

        Must be able to handle FmtStrs because interpreter pass in
        tracebacks already formatted."""
        lines = output.split('\n')
        logger.debug('display_lines: %r', self.display_lines)
        self.current_stdouterr_line += lines[0]
        if len(lines) > 1:
            self.display_lines.extend(paint.display_linize(self.current_stdouterr_line, self.width, blank_line=True))
            self.display_lines.extend(sum((paint.display_linize(line, self.width, blank_line=True) for line in lines[1:-1]), []))
            for line in itertools.chain((self.current_stdouterr_line,), lines[1:-1]):
                if isinstance(line, FmtStr):
                    self.all_logical_lines.append((line.s, LineType.OUTPUT))
                else:
                    self.all_logical_lines.append((line, LineType.OUTPUT))
            self.current_stdouterr_line = lines[-1]
        logger.debug('display_lines: %r', self.display_lines)

    def send_to_stdin(self, line):
        if line.endswith('\n'):
            self.display_lines.extend(paint.display_linize(self.current_output_line, self.width))
            self.current_output_line = ''

    @property
    def done(self):
        """Whether the last block is complete - which prompt to use, ps1 or
        ps2"""
        return not self.buffer

    @property
    def current_line_formatted(self):
        """The colored current line (no prompt, not wrapped)"""
        if self.config.syntax:
            fs = bpythonparse(pygformat(self.tokenize(self.current_line), self.formatter))
            if self.incr_search_mode != SearchMode.NO_SEARCH:
                if self.incr_search_target in self.current_line:
                    fs = fmtfuncs.on_magenta(self.incr_search_target).join(fs.split(self.incr_search_target))
            elif self.rl_history.saved_line and self.rl_history.saved_line in self.current_line:
                if self.config.curtsies_right_arrow_completion and self.rl_history.index != 0:
                    fs = fmtfuncs.on_magenta(self.rl_history.saved_line).join(fs.split(self.rl_history.saved_line))
            logger.debug('Display line %r -> %r', self.current_line, fs)
        else:
            fs = fmtstr(self.current_line)
        if hasattr(self, 'old_fs') and str(fs) != str(self.old_fs):
            pass
        self.old_fs = fs
        return fs

    @property
    def lines_for_display(self):
        """All display lines (wrapped, colored, with prompts)"""
        return self.display_lines + self.display_buffer_lines

    @property
    def display_buffer_lines(self):
        """The display lines (wrapped, colored, +prompts) of current buffer"""
        lines = []
        for display_line in self.display_buffer:
            prompt = func_for_letter(self.config.color_scheme['prompt'])
            more = func_for_letter(self.config.color_scheme['prompt_more'])
            display_line = (more(self.ps2) if lines else prompt(self.ps1)) + display_line
            for line in paint.display_linize(display_line, self.width):
                lines.append(line)
        return lines

    @property
    def display_line_with_prompt(self):
        """colored line with prompt"""
        prompt = func_for_letter(self.config.color_scheme['prompt'])
        more = func_for_letter(self.config.color_scheme['prompt_more'])
        if self.incr_search_mode == SearchMode.REVERSE_INCREMENTAL_SEARCH:
            return prompt(f"(reverse-i-search)`{self.incr_search_target}': ") + self.current_line_formatted
        elif self.incr_search_mode == SearchMode.INCREMENTAL_SEARCH:
            return prompt(f"(i-search)`%s': ") + self.current_line_formatted
        return (prompt(self.ps1) if self.done else more(self.ps2)) + self.current_line_formatted

    @property
    def current_cursor_line_without_suggestion(self):
        """
        Current line, either output/input or Python prompt + code

        :returns: FmtStr
        """
        value = self.current_output_line + ('' if self.coderunner.running else self.display_line_with_prompt)
        logger.debug('current cursor line: %r', value)
        return value

    @property
    def current_cursor_line(self):
        if self.config.curtsies_right_arrow_completion:
            suggest = func_for_letter(self.config.color_scheme['right_arrow_suggestion'])
            return self.current_cursor_line_without_suggestion + suggest(self.current_suggestion)
        else:
            return self.current_cursor_line_without_suggestion

    @property
    def current_suggestion(self):
        if self.current_line:
            for entry in reversed(self.rl_history.entries):
                if entry.startswith(self.current_line):
                    return entry[len(self.current_line):]
        return ''

    @property
    def current_output_line(self):
        """line of output currently being written, and stdin typed"""
        return self.current_stdouterr_line + self.stdin.current_line

    @current_output_line.setter
    def current_output_line(self, value):
        self.current_stdouterr_line = ''
        self.stdin.current_line = '\n'

    def number_of_padding_chars_on_current_cursor_line(self):
        """To avoid cutting off two-column characters at the end of lines where
        there's only one column left, curtsies adds a padding char (u' ').
        It's important to know about these for cursor positioning.

        Should return zero unless there are fullwidth characters."""
        full_line = self.current_cursor_line_without_suggestion
        line_with_padding_len = sum((len(line.s) for line in paint.display_linize(self.current_cursor_line_without_suggestion.s, self.width)))
        return line_with_padding_len - len(full_line)

    def paint(self, about_to_exit=False, user_quit=False, try_preserve_history_height=30, min_infobox_height=5) -> Tuple[FSArray, Tuple[int, int]]:
        """Returns an array of min_height or more rows and width columns, plus
        cursor position

        Paints the entire screen - ideally the terminal display layer will take
        a diff and only write to the screen in portions that have changed, but
        the idea is that we don't need to worry about that here, instead every
        frame is completely redrawn because less state is cool!

        try_preserve_history_height is the the number of rows of content that
        must be visible before the suggestion box scrolls the terminal in order
        to display more than min_infobox_height rows of suggestions, docs etc.
        """
        if about_to_exit:
            self.clean_up_current_line_for_exit()
        width, min_height = (self.width, self.height)
        show_status_bar = (bool(self.status_bar.should_show_message) or self.status_bar.has_focus) and (not self.request_paint_to_pad_bottom)
        if show_status_bar:
            min_height -= 1
        current_line_start_row = len(self.lines_for_display) - max(0, self.scroll_offset)
        if self.request_paint_to_clear_screen:
            self.request_paint_to_clear_screen = False
            arr = FSArray(min_height + current_line_start_row, width)
        elif self.request_paint_to_pad_bottom:
            height = min(self.request_paint_to_pad_bottom, min_height - 1)
            arr = FSArray(height, width)
            self.request_paint_to_pad_bottom = 0
        else:
            arr = FSArray(0, width)
        current_line = paint.paint_current_line(min_height, width, self.current_cursor_line)

        def move_screen_up(current_line_start_row):
            while current_line_start_row < 0:
                logger.debug('scroll_offset was %s, current_line_start_row was %s', self.scroll_offset, current_line_start_row)
                self.scroll_offset = self.scroll_offset - self.height
                current_line_start_row = len(self.lines_for_display) - max(-1, self.scroll_offset)
                logger.debug('scroll_offset changed to %s, current_line_start_row changed to %s', self.scroll_offset, current_line_start_row)
            return current_line_start_row
        if self.inconsistent_history and (not self.history_already_messed_up):
            logger.debug(INCONSISTENT_HISTORY_MSG)
            self.history_already_messed_up = True
            msg = INCONSISTENT_HISTORY_MSG
            arr[0, 0:min(len(msg), width)] = [msg[:width]]
            current_line_start_row += 1
            self.scroll_offset -= 1
            current_line_start_row = move_screen_up(current_line_start_row)
            logger.debug('current_line_start_row: %r', current_line_start_row)
            history = paint.paint_history(max(0, current_line_start_row - 1), width, self.lines_for_display)
            arr[1:history.height + 1, :history.width] = history
            if arr.height <= min_height:
                arr[min_height, 0] = ' '
        elif current_line_start_row < 0:
            logger.debug(CONTIGUITY_BROKEN_MSG)
            msg = CONTIGUITY_BROKEN_MSG
            arr[0, 0:min(len(msg), width)] = [msg[:width]]
            current_line_start_row = move_screen_up(current_line_start_row)
            history = paint.paint_history(max(0, current_line_start_row - 1), width, self.lines_for_display)
            arr[1:history.height + 1, :history.width] = history
            if arr.height <= min_height:
                arr[min_height, 0] = ' '
        else:
            assert current_line_start_row >= 0
            logger.debug('no history issues. start %i', current_line_start_row)
            history = paint.paint_history(current_line_start_row, width, self.lines_for_display)
            arr[:history.height, :history.width] = history
        self.inconsistent_history = False
        if user_quit:
            current_line_start_row = current_line_start_row - current_line.height
        logger.debug('---current line row slice %r, %r', current_line_start_row, current_line_start_row + current_line.height)
        logger.debug('---current line col slice %r, %r', 0, current_line.width)
        arr[current_line_start_row:current_line_start_row + current_line.height, 0:current_line.width] = current_line
        if current_line.height > min_height:
            return (arr, (0, 0))
        lines = paint.display_linize(self.current_cursor_line + 'X', width)
        current_line_end_row = current_line_start_row + len(lines) - 1
        current_line_height = current_line_end_row - current_line_start_row
        if self.stdin.has_focus:
            logger.debug('stdouterr when self.stdin has focus: %r %r', type(self.current_stdouterr_line), self.current_stdouterr_line)
            stdouterr = self.current_stdouterr_line
            if isinstance(stdouterr, FmtStr):
                stdouterr_width = stdouterr.width
            else:
                stdouterr_width = len(stdouterr)
            cursor_row, cursor_column = divmod(stdouterr_width + wcswidth(self.stdin.current_line, max(0, self.stdin.cursor_offset)), width)
            assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, self.current_stdouterr_line, self.stdin.current_line)
        elif self.coderunner.running:
            cursor_row, cursor_column = divmod(len(self.current_cursor_line_without_suggestion) + self.cursor_offset, width)
            assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, len(self.current_cursor_line), len(self.current_line), self.cursor_offset)
        else:
            cursor_row, cursor_column = divmod(wcswidth(self.current_cursor_line_without_suggestion.s) - wcswidth(self.current_line) + wcswidth(self.current_line, max(0, self.cursor_offset)) + self.number_of_padding_chars_on_current_cursor_line(), width)
            assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, self.current_cursor_line_without_suggestion.s, self.current_line, self.cursor_offset)
        cursor_row += current_line_start_row
        if self.list_win_visible and (not self.coderunner.running):
            logger.debug('infobox display code running')
            visible_space_above = history.height
            potential_space_below = min_height - current_line_end_row - 1
            visible_space_below = potential_space_below - self.get_top_usable_line()
            if self.config.curtsies_list_above:
                info_max_rows = max(visible_space_above, visible_space_below)
            else:
                preferred_height = max(min_infobox_height, min_height - try_preserve_history_height)
                info_max_rows = min(max(visible_space_below, preferred_height), min_height - current_line_height - 1)
            infobox = paint.paint_infobox(info_max_rows, int(width * self.config.cli_suggestion_width), self.matches_iter.matches, self.funcprops, self.arg_pos, self.current_match, self.docstring, self.config, self.matches_iter.completer.format if self.matches_iter.completer else None)
            if visible_space_below >= infobox.height or not self.config.curtsies_list_above:
                arr[current_line_end_row + 1:current_line_end_row + 1 + infobox.height, 0:infobox.width] = infobox
            else:
                arr[current_line_start_row - infobox.height:current_line_start_row, 0:infobox.width] = infobox
                logger.debug('infobox of shape %r added to arr of shape %r', infobox.shape, arr.shape)
        logger.debug('about to exit: %r', about_to_exit)
        if show_status_bar:
            statusbar_row = min_height if arr.height == min_height else arr.height
            if about_to_exit:
                arr[statusbar_row, :] = FSArray(1, width)
            else:
                arr[statusbar_row, :] = paint.paint_statusbar(1, width, self.status_bar.current_line, self.config)
        if self.presentation_mode:
            rows = arr.height
            columns = arr.width
            last_key_box = paint.paint_last_events(rows, columns, [events.pp_event(x) for x in self.last_events if x], self.config)
            arr[arr.height - last_key_box.height:arr.height, arr.width - last_key_box.width:arr.width] = last_key_box
        if self.config.color_scheme['background'] not in ('d', 'D'):
            for r in range(arr.height):
                bg = color_for_letter(self.config.color_scheme['background'])
                arr[r] = fmtstr(arr[r], bg=bg)
        logger.debug('returning arr of size %r', arr.shape)
        logger.debug('cursor pos: %r', (cursor_row, cursor_column))
        return (arr, (cursor_row, cursor_column))

    @contextlib.contextmanager
    def in_paste_mode(self):
        orig_value = self.paste_mode
        self.paste_mode = True
        yield
        self.paste_mode = orig_value
        if not self.paste_mode:
            self.update_completion()

    def __repr__(self):
        return f'<{type(self)}\n  cursor_offset: {self.cursor_offset}\n  num display lines: {len(self.display_lines)}\n  lines scrolled down: {self.scroll_offset}\n>'

    def _get_current_line(self) -> str:
        """The current line"""
        return self._current_line

    def _set_current_line(self, line: str, update_completion=True, reset_rl_history=True, clear_special_mode=True):
        if self._current_line == line:
            return
        self._current_line = line
        if self.paste_mode:
            return
        if update_completion:
            self.update_completion()
        if reset_rl_history:
            self.rl_history.reset()
        if clear_special_mode:
            self.special_mode = None
        self.unhighlight_paren()

    def _get_cursor_offset(self) -> int:
        """The current cursor offset from the front of the "line"."""
        return self._cursor_offset

    def _set_cursor_offset(self, offset: int, update_completion=True, reset_rl_history=False, clear_special_mode=True):
        if self._cursor_offset == offset:
            return
        if self.paste_mode:
            self._cursor_offset = offset
            self.unhighlight_paren()
            return
        if reset_rl_history:
            self.rl_history.reset()
        if clear_special_mode:
            self.incr_search_mode = SearchMode.NO_SEARCH
        self._cursor_offset = offset
        if update_completion:
            self.update_completion()
        self.unhighlight_paren()

    def echo(self, msg, redraw=True):
        """
        Notification that redrawing the current line is necessary (we don't
        care, since we always redraw the whole screen)

        Supposed to parse and echo a formatted string with appropriate
        attributes. It's not supposed to update the screen if it's reevaluating
        the code (as it does with undo)."""
        logger.debug('echo called with %r' % msg)

    @property
    def cpos(self):
        """many WATs were had - it's the pos from the end of the line back"""
        return len(self.current_line) - self.cursor_offset

    def reprint_line(self, lineno, tokens):
        logger.debug('calling reprint line with %r %r', lineno, tokens)
        if self.config.syntax:
            self.display_buffer[lineno] = bpythonparse(pygformat(tokens, self.formatter))

    def take_back_buffer_line(self):
        assert len(self.buffer) > 0
        if len(self.buffer) == 1:
            self._cursor_offset = 0
            self.current_line = ''
        else:
            line = self.buffer[-1]
            indent = self.predicted_indent(line)
            self._current_line = indent * ' '
            self.cursor_offset = len(self.current_line)
        self.display_buffer.pop()
        self.buffer.pop()
        self.history.pop()
        self.all_logical_lines.pop()

    def take_back_empty_line(self):
        assert self.history and (not self.history[-1])
        self.history.pop()
        self.display_lines.pop()
        self.all_logical_lines.pop()

    def prompt_undo(self):
        if self.buffer:
            return self.take_back_buffer_line()
        if self.history and (not self.history[-1]):
            return self.take_back_empty_line()

        def prompt_for_undo():
            n = super(BaseRepl, self).prompt_undo()
            if n > 0:
                self.request_undo(n=n)
        greenlet.greenlet(prompt_for_undo).switch()

    def redo(self) -> None:
        if self.redo_stack:
            temp = self.redo_stack.pop()
            self.history.append(temp)
            self.all_logical_lines.append((temp, LineType.INPUT))
            self.push(temp)
        else:
            self.status_bar.message('Nothing to redo.')

    def reevaluate(self, new_code=False):
        """bpython.Repl.undo calls this"""
        if self.watcher:
            self.watcher.reset()
        old_logical_lines = self.history
        old_display_lines = self.display_lines
        self.history = []
        self.display_lines = []
        self.all_logical_lines = []
        if not self.weak_rewind:
            self.interp = self.interp.__class__()
            self.interp.write = self.send_to_stdouterr
            self.coderunner.interp = self.interp
            self.initialize_interp()
        self.buffer = []
        self.display_buffer = []
        self.highlighted_paren = None
        self.process_event(bpythonevents.RunStartupFileEvent())
        self.reevaluating = True
        sys.stdin = ReevaluateFakeStdin(self.stdin, self)
        for line in old_logical_lines:
            self._current_line = line
            self.on_enter(new_code=new_code)
            while self.fake_refresh_requested:
                self.fake_refresh_requested = False
                self.process_event(bpythonevents.RefreshRequestEvent())
        sys.stdin = self.stdin
        self.reevaluating = False
        num_lines_onscreen = len(self.lines_for_display) - max(0, self.scroll_offset)
        display_lines_offscreen = self.display_lines[:len(self.display_lines) - num_lines_onscreen]
        old_display_lines_offscreen = old_display_lines[:len(self.display_lines) - num_lines_onscreen]
        logger.debug('old_display_lines_offscreen %s', '|'.join((str(x) for x in old_display_lines_offscreen)))
        logger.debug('    display_lines_offscreen %s', '|'.join((str(x) for x in display_lines_offscreen)))
        if old_display_lines_offscreen[:len(display_lines_offscreen)] != display_lines_offscreen and (not self.history_already_messed_up):
            self.inconsistent_history = True
        logger.debug('after rewind, self.inconsistent_history is %r', self.inconsistent_history)
        self._cursor_offset = 0
        self.current_line = ''

    def initialize_interp(self) -> None:
        self.coderunner.interp.locals['_repl'] = self
        self.coderunner.interp.runsource('from bpython.curtsiesfrontend._internal import _Helper\n')
        self.coderunner.interp.runsource('help = _Helper(_repl)\n')
        self.coderunner.interp.runsource('del _Helper\n')
        del self.coderunner.interp.locals['_repl']

    def getstdout(self) -> str:
        """
        Returns a string of the current bpython session, wrapped, WITH prompts.
        """
        lines = self.lines_for_display + [self.current_line_formatted]
        s = '\n'.join((x.s if isinstance(x, FmtStr) else x for x in lines)) if lines else ''
        return s

    def focus_on_subprocess(self, args):
        prev_sigwinch_handler = signal.getsignal(signal.SIGWINCH)
        try:
            signal.signal(signal.SIGWINCH, self.orig_sigwinch_handler)
            with Termmode(self.orig_stdin, self.orig_tcattrs):
                terminal = self.window.t
                with terminal.fullscreen():
                    sys.__stdout__.write(terminal.save)
                    sys.__stdout__.write(terminal.move(0, 0))
                    sys.__stdout__.flush()
                    p = subprocess.Popen(args, stdin=self.orig_stdin, stderr=sys.__stderr__, stdout=sys.__stdout__)
                    p.wait()
                    sys.__stdout__.write(terminal.restore)
                    sys.__stdout__.flush()
        finally:
            signal.signal(signal.SIGWINCH, prev_sigwinch_handler)

    def pager(self, text: str) -> None:
        """Runs an external pager on text

        text must be a str"""
        command = get_pager_command()
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(text.encode(getpreferredencoding()))
            tmp.flush()
            self.focus_on_subprocess(command + [tmp.name])

    def show_source(self) -> None:
        try:
            source = self.get_source_of_current_name()
        except SourceNotFound as e:
            self.status_bar.message(f'{e}')
        else:
            if self.config.highlight_show_source:
                source = pygformat(Python3Lexer().get_tokens(source), TerminalFormatter())
            self.pager(source)

    def help_text(self) -> str:
        return self.version_help_text() + '\n' + self.key_help_text()

    def version_help_text(self) -> str:
        help_message = _("\nThanks for using bpython!\n\nSee http://bpython-interpreter.org/ for more information and http://docs.bpython-interpreter.org/ for docs.\nPlease report issues at https://github.com/bpython/bpython/issues\n\nFeatures:\nTry using undo ({config.undo_key})!\nEdit the current line ({config.edit_current_block_key}) or the entire session ({config.external_editor_key}) in an external editor. (currently {config.editor})\nSave sessions ({config.save_key}) or post them to pastebins ({config.pastebin_key})! Current pastebin helper: {config.pastebin_helper}\nReload all modules and rerun session ({config.reimport_key}) to test out changes to a module.\nToggle auto-reload mode ({config.toggle_file_watch_key}) to re-execute the current session when a module you've imported is modified.\n\nbpython -i your_script.py runs a file in interactive mode\nbpython -t your_script.py pastes the contents of a file into the session\n\nA config file at {config.config_path} customizes keys and behavior of bpython.\nYou can also set which pastebin helper and which external editor to use.\nSee {example_config_url} for an example config file.\nPress {config.edit_config_key} to edit this config file.\n").format(example_config_url=EXAMPLE_CONFIG_URL, config=self.config)
        return f'bpython-curtsies version {__version__} using curtsies version {curtsies_version}\n{help_message}'

    def key_help_text(self) -> str:
        NOT_IMPLEMENTED = ('suspend', 'cut to buffer', 'search', 'last output', 'yank from buffer', 'cut to buffer')
        pairs = [['complete history suggestion', 'right arrow at end of line'], ['previous match with current line', 'up arrow']]
        for functionality, key in ((attr[:-4].replace('_', ' '), getattr(self.config, attr)) for attr in self.config.__dict__ if attr.endswith('key')):
            if functionality in NOT_IMPLEMENTED:
                key = 'Not Implemented'
            if key == '':
                key = 'Disabled'
            pairs.append([functionality, key])
        max_func = max((len(func) for func, key in pairs))
        return '\n'.join((f'{func.rjust(max_func)} : {key}' for func, key in pairs))

    def get_session_formatted_for_file(self) -> str:

        def process():
            for line, lineType in self.all_logical_lines:
                if lineType == LineType.INPUT:
                    yield line
                elif line.rstrip():
                    yield ('# OUT: %s' % line)
            yield ('### %s' % self.current_line)
        return '\n'.join(process())

    @property
    def ps1(self):
        return _process_ps(super().ps1, '>>> ')

    @property
    def ps2(self):
        return _process_ps(super().ps2, '... ')