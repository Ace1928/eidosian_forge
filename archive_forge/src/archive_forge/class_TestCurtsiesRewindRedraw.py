import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
class TestCurtsiesRewindRedraw(HigherLevelCurtsiesPaintingTest):

    def test_rewind(self):
        self.repl.current_line = '1 + 1'
        self.enter()
        screen = ['>>> 1 + 1', '2', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (2, 4))
        self.repl.undo()
        screen = ['>>> ']
        self.assert_paint_ignoring_formatting(screen, (0, 4))

    def test_rewind_contiguity_loss(self):
        self.enter('1 + 1')
        self.enter('2 + 2')
        self.enter('def foo(x):')
        self.repl.current_line = '    return x + 1'
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> def foo(x):', '...     return x + 1']
        self.assert_paint_ignoring_formatting(screen, (5, 8))
        self.repl.scroll_offset = 1
        self.assert_paint_ignoring_formatting(screen[1:], (4, 8))
        self.undo()
        screen = ['2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (3, 4))
        self.undo()
        screen = ['2', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (1, 4))
        self.undo()
        screen = [CONTIGUITY_BROKEN_MSG[:self.repl.width], '>>> ', '', '', '', ' ']
        self.assert_paint_ignoring_formatting(screen, (1, 4))
        screen = ['>>> ']
        self.assert_paint_ignoring_formatting(screen, (0, 4))

    def test_inconsistent_history_doesnt_happen_if_onscreen(self):
        self.enter('1 + 1')
        screen = ['>>> 1 + 1', '2', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (2, 4))
        self.enter('2 + 2')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))
        self.repl.display_lines[0] = self.repl.display_lines[0] * 2
        self.undo()
        screen = ['>>> 1 + 1', '2', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (2, 4))

    def test_rewind_inconsistent_history(self):
        self.enter('1 + 1')
        self.enter('2 + 2')
        self.enter('3 + 3')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> 3 + 3', '6', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (6, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[2:], (4, 4))
        self.repl.display_lines[0] = self.repl.display_lines[0] * 2
        self.undo()
        screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '>>> 2 + 2', '4', '>>> ', '', ' ']
        self.assert_paint_ignoring_formatting(screen, (3, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[1:-2], (2, 4))
        self.assert_paint_ignoring_formatting(screen[1:-2], (2, 4))

    def test_rewind_inconsistent_history_more_lines_same_screen(self):
        self.repl.width = 60
        sys.a = 5
        self.enter('import sys')
        self.enter('for i in range(sys.a):')
        self.enter('    print(sys.a)')
        self.enter('')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> import sys', '>>> for i in range(sys.a):', '...     print(sys.a)', '... ', '5', '5', '5', '5', '5', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (13, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[9:], (4, 4))
        sys.a = 6
        self.undo()
        screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '6', '>>> 1 + 1', '2', '>>> ', ' ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[1:-1], (3, 4))

    def test_rewind_inconsistent_history_more_lines_lower_screen(self):
        self.repl.width = 60
        sys.a = 5
        self.enter('import sys')
        self.enter('for i in range(sys.a):')
        self.enter('    print(sys.a)')
        self.enter('')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> import sys', '>>> for i in range(sys.a):', '...     print(sys.a)', '... ', '5', '5', '5', '5', '5', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (13, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[9:], (4, 4))
        sys.a = 8
        self.undo()
        screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '8', '8', '8', '>>> 1 + 1', '2', '>>> ']
        self.assert_paint_ignoring_formatting(screen)
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[-5:])

    def test_rewind_inconsistent_history_more_lines_raise_screen(self):
        self.repl.width = 60
        sys.a = 5
        self.enter('import sys')
        self.enter('for i in range(sys.a):')
        self.enter('    print(sys.a)')
        self.enter('')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> import sys', '>>> for i in range(sys.a):', '...     print(sys.a)', '... ', '5', '5', '5', '5', '5', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (13, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[9:], (4, 4))
        sys.a = 1
        self.undo()
        screen = [INCONSISTENT_HISTORY_MSG[:self.repl.width], '1', '>>> 1 + 1', '2', '>>> ', ' ']
        self.assert_paint_ignoring_formatting(screen)
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[1:-1])

    def test_rewind_history_not_quite_inconsistent(self):
        self.repl.width = 50
        sys.a = 5
        self.enter("for i in range(__import__('sys').a):")
        self.enter('    print(i)')
        self.enter('')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = [">>> for i in range(__import__('sys').a):", '...     print(i)', '... ', '0', '1', '2', '3', '4', '>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (12, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[8:], (4, 4))
        sys.a = 6
        self.undo()
        screen = ['5', '>>> 1 + 1', '2', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (3, 4))

    def test_rewind_barely_consistent(self):
        self.enter('1 + 1')
        self.enter('2 + 2')
        self.enter('3 + 3')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> 3 + 3', '6', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (6, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[2:], (4, 4))
        self.repl.display_lines[2] = self.repl.display_lines[2] * 2
        self.undo()
        screen = ['>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (2, 4))

    def test_clear_screen(self):
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))
        self.repl.request_paint_to_clear_screen = True
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ', '', '', '', '']
        self.assert_paint_ignoring_formatting(screen, (4, 4))

    def test_scroll_down_while_banner_visible(self):
        self.repl.status_bar.message('STATUS_BAR')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ', 'STATUS_BAR                      ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[1:], (3, 4))

    def test_clear_screen_while_banner_visible(self):
        self.repl.status_bar.message('STATUS_BAR')
        self.enter('1 + 1')
        self.enter('2 + 2')
        screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ', 'STATUS_BAR                      ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))
        self.repl.scroll_offset += len(screen) - self.repl.height
        self.assert_paint_ignoring_formatting(screen[1:], (3, 4))
        self.repl.request_paint_to_clear_screen = True
        screen = ['2', '>>> 2 + 2', '4', '>>> ', '', '', '', 'STATUS_BAR                      ']
        self.assert_paint_ignoring_formatting(screen, (3, 4))

    def test_cursor_stays_at_bottom_of_screen(self):
        """infobox showing up during intermediate render was causing this to
        fail, #371"""
        self.repl.width = 50
        self.repl.current_line = "__import__('random').__name__"
        with output_to_repl(self.repl):
            self.repl.on_enter(new_code=False)
        screen = [">>> __import__('random').__name__", "'random'"]
        self.assert_paint_ignoring_formatting(screen)
        with output_to_repl(self.repl):
            self.repl.process_event(self.refresh_requests.pop())
        screen = [">>> __import__('random').__name__", "'random'", '']
        self.assert_paint_ignoring_formatting(screen)
        with output_to_repl(self.repl):
            self.repl.process_event(self.refresh_requests.pop())
        screen = [">>> __import__('random').__name__", "'random'", '>>> ']
        self.assert_paint_ignoring_formatting(screen, (2, 4))

    def test_unhighlight_paren_bugs(self):
        """two previous bugs, parent didn't highlight until next render
        and paren didn't unhighlight until enter"""
        self.repl.width = 32
        self.assertEqual(self.repl.rl_history.entries, [''])
        self.enter('(')
        self.assertEqual(self.repl.rl_history.entries, [''])
        screen = ['>>> (', '... ']
        self.assertEqual(self.repl.rl_history.entries, [''])
        self.assert_paint_ignoring_formatting(screen)
        self.assertEqual(self.repl.rl_history.entries, [''])
        with output_to_repl(self.repl):
            self.assertEqual(self.repl.rl_history.entries, [''])
            self.repl.process_event(')')
            self.assertEqual(self.repl.rl_history.entries, [''])
        screen = fsarray([cyan('>>> ') + on_magenta(bold(red('('))), green('... ') + on_magenta(bold(red(')')))], width=32)
        self.assert_paint(screen, (1, 5))
        with output_to_repl(self.repl):
            self.repl.process_event(' ')
        screen = fsarray([cyan('>>> ') + yellow('('), green('... ') + yellow(')') + bold(cyan(' '))], width=32)
        self.assert_paint(screen, (1, 6))

    def test_472(self):
        [self.send_key(c) for c in '(1, 2, 3)']
        with output_to_repl(self.repl):
            self.send_key('\n')
            self.send_refreshes()
            self.send_key('<UP>')
            self.repl.paint()
            [self.send_key('<LEFT>') for _ in range(4)]
            self.send_key('<BACKSPACE>')
            self.send_key('4')
            self.repl.on_enter()
            self.send_refreshes()
        screen = ['>>> (1, 2, 3)', '(1, 2, 3)', '>>> (1, 4, 3)', '(1, 4, 3)', '>>> ']
        self.assert_paint_ignoring_formatting(screen, (4, 4))