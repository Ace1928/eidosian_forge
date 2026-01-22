from __future__ import annotations
import atexit
import copy
import errno
import fcntl
import os
import pty
import selectors
import signal
import struct
import sys
import termios
import time
import traceback
import typing
import warnings
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from urwid import event_loop, util
from urwid.canvas import Canvas
from urwid.display import AttrSpec, RealTerminal
from urwid.display.escape import ALT_DEC_SPECIAL_CHARS, DEC_SPECIAL_CHARS
from urwid.widget import Sizing, Widget
from .display.common import _BASIC_COLORS, _color_desc_256, _color_desc_true
class TermCanvas(Canvas):
    cacheable = False

    def __init__(self, width: int, height: int, widget: Terminal) -> None:
        super().__init__()
        self.width, self.height = (width, height)
        self.widget = widget
        self.modes: TermModes = widget.term_modes
        self.has_focus = False
        self.scrollback_buffer: deque[list[tuple[AttrSpec | None, str | None, bytes]]] = deque(maxlen=10000)
        self.scrolling_up = 0
        self.utf8_eat_bytes: int | None = None
        self.utf8_buffer = bytearray()
        self.escbuf = b''
        self.coords['cursor'] = (0, 0, None)
        self.term_cursor: tuple[int, int] = (0, 0)
        self.within_escape = False
        self.parsestate = 0
        self.attrspec: AttrSpec | None = None
        self.charset = TermCharset()
        self.saved_cursor: tuple[int, int] | None = None
        self.saved_attrs: tuple[AttrSpec | None, TermCharset] | None = None
        self.is_rotten_cursor = False
        self.scrollregion_start = 0
        self.scrollregion_end = self.height - 1
        self.tabstops: list[int] = []
        self.term: list[list[tuple[AttrSpec | None, str | None, bytes]]] = []
        self.reset()

    def set_term_cursor(self, x: int | None=None, y: int | None=None) -> None:
        """
        Set terminal cursor to x/y and update canvas cursor. If one or both axes
        are omitted, use the values of the current position.
        """
        if x is None:
            x = self.term_cursor[0]
        if y is None:
            y = self.term_cursor[1]
        self.term_cursor = self.constrain_coords(x, y)
        if self.has_focus and self.modes.visible_cursor and (self.scrolling_up < self.height - y):
            self.cursor = (x, y + self.scrolling_up)
        else:
            self.cursor = None

    def reset_scroll(self) -> None:
        """
        Reset scrolling region to full terminal size.
        """
        self.scrollregion_start = 0
        self.scrollregion_end = self.height - 1

    def scroll_buffer(self, up: bool=True, reset: bool=False, lines: int | None=None) -> None:
        """
        Scroll the scrolling buffer up (up=True) or down (up=False) the given
        amount of lines or half the screen height.

        If just 'reset' is True, set the scrollbuffer view to the current
        terminal content.
        """
        if reset:
            self.scrolling_up = 0
            self.set_term_cursor()
            return
        if lines is None:
            lines = self.height // 2
        if not up:
            lines = -lines
        maxscroll = len(self.scrollback_buffer)
        self.scrolling_up += lines
        if self.scrolling_up > maxscroll:
            self.scrolling_up = maxscroll
        elif self.scrolling_up < 0:
            self.scrolling_up = 0
        self.set_term_cursor()

    def reset(self) -> None:
        """
        Reset the terminal.
        """
        self.escbuf = b''
        self.within_escape = False
        self.parsestate = 0
        self.attrspec = None
        self.charset = TermCharset()
        self.saved_cursor = None
        self.saved_attrs = None
        self.is_rotten_cursor = False
        self.reset_scroll()
        self.init_tabstops()
        self.modes.reset()
        self.clear()

    def init_tabstops(self, extend: bool=False) -> None:
        tablen, mod = divmod(self.width, 8)
        if mod > 0:
            tablen += 1
        if extend:
            while len(self.tabstops) < tablen:
                self.tabstops.append(1 << 0)
        else:
            self.tabstops = [1 << 0] * tablen

    def set_tabstop(self, x: int | None=None, remove: bool=False, clear: bool=False) -> None:
        if clear:
            for tab in range(len(self.tabstops)):
                self.tabstops[tab] = 0
            return
        if x is None:
            x = self.term_cursor[0]
        div, mod = divmod(x, 8)
        if remove:
            self.tabstops[div] &= ~(1 << mod)
        else:
            self.tabstops[div] |= 1 << mod

    def is_tabstop(self, x: int | None=None) -> bool:
        if x is None:
            x = self.term_cursor[0]
        div, mod = divmod(x, 8)
        return self.tabstops[div] & 1 << mod > 0

    def empty_line(self, char: bytes=b' ') -> list[tuple[AttrSpec | None, str | None, bytes]]:
        return [self.empty_char(char)] * self.width

    def empty_char(self, char: bytes=b' ') -> tuple[AttrSpec | None, str | None, bytes]:
        return (self.attrspec, self.charset.current, char)

    def addstr(self, data: Iterable[int]) -> None:
        if self.width <= 0 or self.height <= 0:
            return
        for byte in data:
            self.addbyte(byte)

    def resize(self, width: int, height: int) -> None:
        """
        Resize the terminal to the given width and height.
        """
        x, y = self.term_cursor
        if width > self.width:
            for y in range(self.height):
                self.term[y] += [self.empty_char()] * (width - self.width)
        elif width < self.width:
            for y in range(self.height):
                self.term[y] = self.term[y][:width]
        self.width = width
        if height > self.height:
            for _y in range(self.height, height):
                try:
                    last_line = self.scrollback_buffer.pop()
                except IndexError:
                    self.term.append(self.empty_line())
                    self.scrollregion_end += 1
                    continue
                padding = self.width - len(last_line)
                if padding > 0:
                    last_line += [self.empty_char()] * padding
                else:
                    last_line = last_line[:self.width]
                self.term.insert(0, last_line)
        elif height < self.height:
            for _y in range(height, self.height):
                self.scrollback_buffer.append(self.term.pop(0))
        self.height = height
        self.reset_scroll()
        x, y = self.constrain_coords(x, y)
        self.set_term_cursor(x, y)
        self.init_tabstops(extend=True)

    def set_g01(self, char: bytes, mod: bytes) -> None:
        """
        Set G0 or G1 according to 'char' and modifier 'mod'.
        """
        if self.modes.main_charset != CHARSET_DEFAULT:
            return
        if mod == b'(':
            g = 0
        else:
            g = 1
        if char == b'0':
            cset = 'vt100'
        elif char == b'U':
            cset = 'ibmpc'
        elif char == b'K':
            cset = 'user'
        else:
            cset = 'default'
        self.charset.define(g, cset)

    def parse_csi(self, char: bytes) -> None:
        """
        Parse ECMA-48 CSI (Control Sequence Introducer) sequences.
        """
        qmark = self.escbuf.startswith(b'?')
        escbuf = []
        for arg in self.escbuf[1 if qmark else 0:].split(b';'):
            try:
                num = int(arg)
            except ValueError:
                num = None
            escbuf.append(num)
        cmd_ = CSI_COMMANDS[char]
        if cmd_ is not None:
            if isinstance(cmd_, CSIAlias):
                csi_cmd: CSICommand = CSI_COMMANDS[cmd_.alias]
            elif isinstance(cmd_, CSICommand):
                csi_cmd = cmd_
            elif cmd_[0] == 'alias':
                csi_cmd = CSI_COMMANDS[CSIAlias(*cmd_).alias]
            else:
                csi_cmd = CSICommand(*cmd_)
            number_of_args, default_value, cmd = csi_cmd
            while len(escbuf) < number_of_args:
                escbuf.append(default_value)
            for i in range(len(escbuf)):
                if escbuf[i] is None or escbuf[i] == 0:
                    escbuf[i] = default_value
            with suppress(ValueError):
                cmd(self, escbuf, qmark)

    def parse_noncsi(self, char: bytes, mod: bytes=b'') -> None:
        """
        Parse escape sequences which are not CSI.
        """
        if mod == b'#' and char == b'8':
            self.decaln()
        elif mod == b'%':
            if char == b'@':
                self.modes.main_charset = CHARSET_DEFAULT
            elif char in b'G8':
                self.modes.main_charset = CHARSET_UTF8
        elif mod in {b'(', b')'}:
            self.set_g01(char, mod)
        elif char == b'M':
            self.linefeed(reverse=True)
        elif char == b'D':
            self.linefeed()
        elif char == b'c':
            self.reset()
        elif char == b'E':
            self.newline()
        elif char == b'H':
            self.set_tabstop()
        elif char == b'Z':
            self.widget.respond(f'{ESC}[?6c')
        elif char == b'7':
            self.save_cursor(with_attrs=True)
        elif char == b'8':
            self.restore_cursor(with_attrs=True)

    def parse_osc(self, buf: bytes) -> None:
        """
        Parse operating system command.
        """
        if buf.startswith((b';', b'0;', b'2;')):
            self.widget.set_title(buf.decode().partition(';')[2])

    def parse_escape(self, char: bytes) -> None:
        if self.parsestate == 1:
            if char in CSI_COMMANDS:
                self.parse_csi(char)
                self.parsestate = 0
            elif char in b'0123456789;' or (not self.escbuf and char == b'?'):
                self.escbuf += char
                return
        elif self.parsestate == 0 and char == b']':
            self.escbuf = b''
            self.parsestate = 2
            return
        elif self.parsestate == 2 and char == b'\x07':
            self.parse_osc(self.escbuf.lstrip(b'0'))
        elif self.parsestate == 2 and self.escbuf[-1:] + char == f'{ESC}\\'.encode('iso8859-1'):
            self.parse_osc(self.escbuf[:-1].lstrip(b'0'))
        elif self.parsestate == 2 and self.escbuf.startswith(b'P') and (len(self.escbuf) == 8):
            pass
        elif self.parsestate == 2 and (not self.escbuf) and (char == b'R'):
            pass
        elif self.parsestate == 2:
            self.escbuf += char
            return
        elif self.parsestate == 0 and char == b'[':
            self.escbuf = b''
            self.parsestate = 1
            return
        elif self.parsestate == 0 and char in {b'%', b'#', b'(', b')'}:
            self.escbuf = char
            self.parsestate = 3
            return
        elif self.parsestate == 3:
            self.parse_noncsi(char, self.escbuf)
        elif char in {b'c', b'D', b'E', b'H', b'M', b'Z', b'7', b'8', b'>', b'='}:
            self.parse_noncsi(char)
        self.leave_escape()

    def leave_escape(self) -> None:
        self.within_escape = False
        self.parsestate = 0
        self.escbuf = b''

    def get_utf8_len(self, bytenum: int) -> int:
        """
        Process startbyte and return the number of bytes following it to get a
        valid UTF-8 multibyte sequence.

        bytenum -- an integer ordinal
        """
        length = 0
        while bytenum & 64:
            bytenum <<= 1
            length += 1
        return length

    def addbyte(self, byte: int) -> None:
        """
        Parse main charset and add the processed byte(s) to the terminal state
        machine.

        byte -- an integer ordinal
        """
        if self.modes.main_charset == CHARSET_UTF8 or util.get_encoding() == 'utf8':
            if byte >= 192:
                self.utf8_eat_bytes = self.get_utf8_len(byte)
                self.utf8_buffer = bytearray([byte])
                return
            if 128 <= byte < 192 and self.utf8_eat_bytes is not None:
                if self.utf8_eat_bytes > 1:
                    self.utf8_eat_bytes -= 1
                    self.utf8_buffer.append(byte)
                    return
                self.utf8_eat_bytes = None
                sequence = (self.utf8_buffer + bytes([byte])).decode('utf-8', 'ignore')
                if not sequence:
                    return
                char = sequence.encode(util.get_encoding(), 'replace')
            else:
                self.utf8_eat_bytes = None
                char = bytes([byte])
        else:
            char = bytes([byte])
        self.process_char(char)

    def process_char(self, char: int | bytes) -> None:
        """
        Process a single character (single- and multi-byte).

        char -- a byte string
        """
        x, y = self.term_cursor
        if isinstance(char, int):
            char = char.to_bytes(1, 'little')
        dc = self.modes.display_ctrl
        if char == ESC_B and self.parsestate != 2:
            self.within_escape = True
        elif not dc and char == b'\r':
            self.carriage_return()
        elif not dc and char == b'\x0f':
            self.charset.activate(0)
        elif not dc and char == b'\x0e':
            self.charset.activate(1)
        elif not dc and char in b'\n\x0b\x0c':
            self.linefeed()
            if self.modes.lfnl:
                self.carriage_return()
        elif not dc and char == b'\t':
            self.tab()
        elif not dc and char == b'\x08':
            if x > 0:
                self.set_term_cursor(x - 1, y)
        elif not dc and char == b'\x07' and (self.parsestate != 2):
            self.widget.beep()
        elif not dc and char in b'\x18\x1a':
            self.leave_escape()
        elif not dc and char in b'\x00\x7f':
            pass
        elif self.within_escape:
            self.parse_escape(char)
        elif not dc and char == b'\x9b':
            self.within_escape = True
            self.escbuf = b''
            self.parsestate = 1
        else:
            self.push_cursor(char)

    def set_char(self, char: bytes, x: int | None=None, y: int | None=None) -> None:
        """
        Set character of either the current cursor position
        or a position given by 'x' and/or 'y' to 'char'.
        """
        if x is None:
            x = self.term_cursor[0]
        if y is None:
            y = self.term_cursor[1]
        x, y = self.constrain_coords(x, y)
        self.term[y][x] = (self.attrspec, self.charset.current, char)

    def constrain_coords(self, x: int, y: int, ignore_scrolling: bool=False) -> tuple[int, int]:
        """
        Checks if x/y are within the terminal and returns the corrected version.
        If 'ignore_scrolling' is set, constrain within the full size of the
        screen and not within scrolling region.
        """
        if x >= self.width:
            x = self.width - 1
        elif x < 0:
            x = 0
        if self.modes.constrain_scrolling and (not ignore_scrolling):
            if y > self.scrollregion_end:
                y = self.scrollregion_end
            elif y < self.scrollregion_start:
                y = self.scrollregion_start
        elif y >= self.height:
            y = self.height - 1
        elif y < 0:
            y = 0
        return (x, y)

    def linefeed(self, reverse: bool=False) -> None:
        """
        Move the cursor down (or up if reverse is True) one line but don't reset
        horizontal position.
        """
        x, y = self.term_cursor
        if reverse:
            if y <= 0 < self.scrollregion_start:
                pass
            elif y == self.scrollregion_start:
                self.scroll(reverse=True)
            else:
                y -= 1
        elif y >= self.height - 1 > self.scrollregion_end:
            pass
        elif y == self.scrollregion_end:
            self.scroll()
        else:
            y += 1
        self.set_term_cursor(x, y)

    def carriage_return(self) -> None:
        self.set_term_cursor(0, self.term_cursor[1])

    def newline(self) -> None:
        """
        Do a carriage return followed by a line feed.
        """
        self.carriage_return()
        self.linefeed()

    def move_cursor(self, x: int, y: int, relative_x: bool=False, relative_y: bool=False, relative: bool=False) -> None:
        """
        Move cursor to position x/y while constraining terminal sizes.
        If 'relative' is True, x/y is relative to the current cursor
        position. 'relative_x' and 'relative_y' is the same but just with
        the corresponding axis.
        """
        if relative:
            relative_y = relative_x = True
        if relative_x:
            x += self.term_cursor[0]
        if relative_y:
            y += self.term_cursor[1]
        elif self.modes.constrain_scrolling:
            y += self.scrollregion_start
        self.set_term_cursor(x, y)

    def push_char(self, char: bytes | None, x: int, y: int) -> None:
        """
        Push one character to current position and advance cursor to x/y.
        """
        if char is not None:
            char = self.charset.apply_mapping(char)
            if self.modes.insert:
                self.insert_chars(char=char)
            else:
                self.set_char(char)
        self.set_term_cursor(x, y)

    def push_cursor(self, char: bytes | None=None) -> None:
        """
        Move cursor one character forward wrapping lines as needed.
        If 'char' is given, put the character into the former position.
        """
        x, y = self.term_cursor
        if self.modes.autowrap:
            if x + 1 >= self.width and (not self.is_rotten_cursor):
                self.is_rotten_cursor = True
                self.push_char(char, x, y)
            else:
                x += 1
                if x >= self.width and self.is_rotten_cursor:
                    if y >= self.scrollregion_end:
                        self.scroll()
                    else:
                        y += 1
                    x = 1
                    self.set_term_cursor(0, y)
                self.push_char(char, x, y)
                self.is_rotten_cursor = False
        else:
            if x + 1 < self.width:
                x += 1
            self.is_rotten_cursor = False
            self.push_char(char, x, y)

    def save_cursor(self, with_attrs: bool=False) -> None:
        self.saved_cursor = tuple(self.term_cursor)
        if with_attrs:
            self.saved_attrs = (copy.copy(self.attrspec), copy.copy(self.charset))

    def restore_cursor(self, with_attrs: bool=False) -> None:
        if self.saved_cursor is None:
            return
        x, y = self.saved_cursor
        self.set_term_cursor(x, y)
        if with_attrs and self.saved_attrs is not None:
            self.attrspec, self.charset = (copy.copy(self.saved_attrs[0]), copy.copy(self.saved_attrs[1]))

    def tab(self, tabstop: int=8) -> None:
        """
        Moves cursor to the next 'tabstop' filling everything in between
        with spaces.
        """
        x, y = self.term_cursor
        while x < self.width - 1:
            self.set_char(b' ')
            x += 1
            if self.is_tabstop(x):
                break
        self.is_rotten_cursor = False
        self.set_term_cursor(x, y)

    def scroll(self, reverse: bool=False) -> None:
        """
        Append a new line at the bottom and put the topmost line into the
        scrollback buffer.

        If reverse is True, do exactly the opposite, but don't save into
        scrollback buffer.
        """
        if reverse:
            self.term.pop(self.scrollregion_end)
            self.term.insert(self.scrollregion_start, self.empty_line())
        else:
            killed = self.term.pop(self.scrollregion_start)
            self.scrollback_buffer.append(killed)
            self.term.insert(self.scrollregion_end, self.empty_line())

    def decaln(self) -> None:
        """
        DEC screen alignment test: Fill screen with E's.
        """
        for row in range(self.height):
            self.term[row] = self.empty_line(b'E')

    def blank_line(self, row: int) -> None:
        """
        Blank a single line at the specified row, without modifying other lines.
        """
        self.term[row] = self.empty_line()

    def insert_chars(self, position: tuple[int, int] | None=None, chars: int=1, char: bytes | None=None) -> None:
        """
        Insert 'chars' number of either empty characters - or those specified by
        'char' - before 'position' (or the current position if not specified)
        pushing subsequent characters of the line to the right without wrapping.
        """
        if position is None:
            position = self.term_cursor
        if chars == 0:
            chars = 1
        if char is None:
            char_spec = self.empty_char()
        else:
            char_spec = (self.attrspec, self.charset.current, char)
        x, y = position
        while chars > 0:
            self.term[y].insert(x, char_spec)
            self.term[y].pop()
            chars -= 1

    def remove_chars(self, position: tuple[int, int] | None=None, chars: int=1) -> None:
        """
        Remove 'chars' number of empty characters from 'position' (or the current
        position if not specified) pulling subsequent characters of the line to
        the left without joining any subsequent lines.
        """
        if position is None:
            position = self.term_cursor
        if chars == 0:
            chars = 1
        x, y = position
        while chars > 0:
            self.term[y].pop(x)
            self.term[y].append(self.empty_char())
            chars -= 1

    def insert_lines(self, row: int | None=None, lines: int=1) -> None:
        """
        Insert 'lines' of empty lines after the specified row, pushing all
        subsequent lines to the bottom. If no 'row' is specified, the current
        row is used.
        """
        if row is None:
            row = self.term_cursor[1]
        else:
            row = self.scrollregion_start
        if lines == 0:
            lines = 1
        while lines > 0:
            self.term.insert(row, self.empty_line())
            self.term.pop(self.scrollregion_end)
            lines -= 1

    def remove_lines(self, row: int | None=None, lines: int=1) -> None:
        """
        Remove 'lines' number of lines at the specified row, pulling all
        subsequent lines to the top. If no 'row' is specified, the current row
        is used.
        """
        if row is None:
            row = self.term_cursor[1]
        else:
            row = self.scrollregion_start
        if lines == 0:
            lines = 1
        while lines > 0:
            self.term.pop(row)
            self.term.insert(self.scrollregion_end, self.empty_line())
            lines -= 1

    def erase(self, start: tuple[int, int] | tuple[int, int, bool], end: tuple[int, int] | tuple[int, int, bool]) -> None:
        """
        Erase a region of the terminal. The 'start' tuple (x, y) defines the
        starting position of the erase, while end (x, y) the last position.

        For example if the terminal size is 4x3, start=(1, 1) and end=(1, 2)
        would erase the following region:

        ....
        .XXX
        XX..
        """
        sx, sy = self.constrain_coords(*start)
        ex, ey = self.constrain_coords(*end)
        if sy == ey:
            for x in range(sx, ex + 1):
                self.term[sy][x] = self.empty_char()
            return
        y = sy
        while y <= ey:
            if y == sy:
                for x in range(sx, self.width):
                    self.term[y][x] = self.empty_char()
            elif y == ey:
                for x in range(ex + 1):
                    self.term[y][x] = self.empty_char()
            else:
                self.blank_line(y)
            y += 1

    def sgi_to_attrspec(self, attrs: Sequence[int], fg: int, bg: int, attributes: set[str], prev_colors: int) -> AttrSpec | None:
        """
        Parse SGI sequence and return an AttrSpec representing the sequence
        including all earlier sequences specified as 'fg', 'bg' and
        'attributes'.
        """
        idx = 0
        colors = prev_colors
        while idx < len(attrs):
            attr = attrs[idx]
            if 30 <= attr <= 37:
                fg = attr - 30
                colors = max(16, colors)
            elif 40 <= attr <= 47:
                bg = attr - 40
                colors = max(16, colors)
            elif attr in {38, 48}:
                if idx + 2 < len(attrs) and attrs[idx + 1] == 5:
                    color = attrs[idx + 2]
                    colors = max(256, colors)
                    if attr == 38:
                        fg = color
                    else:
                        bg = color
                    idx += 2
                elif idx + 4 < len(attrs) and attrs[idx + 1] == 2:
                    color = (attrs[idx + 2] << 16) + (attrs[idx + 3] << 8) + attrs[idx + 4]
                    colors = 2 ** 24
                    if attr == 38:
                        fg = color
                    else:
                        bg = color
                    idx += 4
            elif attr == 39:
                fg = None
            elif attr == 49:
                bg = None
            elif attr == 10:
                self.charset.reset_sgr_ibmpc()
                self.modes.display_ctrl = False
            elif attr in {11, 12}:
                self.charset.set_sgr_ibmpc()
                self.modes.display_ctrl = True
            elif attr == 1:
                attributes.add('bold')
            elif attr == 4:
                attributes.add('underline')
            elif attr == 5:
                attributes.add('blink')
            elif attr == 7:
                attributes.add('standout')
            elif attr == 24:
                attributes.discard('underline')
            elif attr == 25:
                attributes.discard('blink')
            elif attr == 27:
                attributes.discard('standout')
            elif attr == 0:
                fg = bg = None
                attributes.clear()
            idx += 1
        if 'bold' in attributes and colors == 16 and (fg is not None) and (fg < 8):
            fg += 8

        def _defaulter(color: int | None, colors: int) -> str:
            if color is None:
                return 'default'
            if color > 255 or colors == 2 ** 24:
                return _color_desc_true(color)
            if color > 15 or colors == 256:
                return _color_desc_256(color)
            return _BASIC_COLORS[color]
        decoded_fg = _defaulter(fg, colors)
        decoded_bg = _defaulter(bg, colors)
        if attributes:
            decoded_fg = ','.join((decoded_fg, *list(attributes)))
        if decoded_fg == decoded_bg == 'default':
            return None
        if colors:
            return AttrSpec(decoded_fg, decoded_bg, colors=colors)
        return AttrSpec(decoded_fg, decoded_bg)

    def csi_set_attr(self, attrs: Sequence[int]) -> None:
        """
        Set graphics rendition.
        """
        if attrs[-1] == 0:
            self.attrspec = None
        attributes = set()
        if self.attrspec is None:
            fg = bg = None
        else:
            if 'default' in self.attrspec.foreground:
                fg = None
            else:
                fg = self.attrspec.foreground_number
                if fg >= 8 and self.attrspec.colors == 16:
                    fg -= 8
            if 'default' in self.attrspec.background:
                bg = None
            else:
                bg = self.attrspec.background_number
                if bg >= 8 and self.attrspec.colors == 16:
                    bg -= 8
            for attr in ('bold', 'underline', 'blink', 'standout'):
                if not getattr(self.attrspec, attr):
                    continue
                attributes.add(attr)
        attrspec = self.sgi_to_attrspec(attrs, fg, bg, attributes, self.attrspec.colors if self.attrspec else 1)
        if self.modes.reverse_video:
            self.attrspec = self.reverse_attrspec(attrspec)
        else:
            self.attrspec = attrspec

    def reverse_attrspec(self, attrspec: AttrSpec | None, undo: bool=False) -> AttrSpec:
        """
        Put standout mode to the 'attrspec' given and remove it if 'undo' is
        True.
        """
        if attrspec is None:
            attrspec = AttrSpec('default', 'default')
        attrs = [fg.strip() for fg in attrspec.foreground.split(',')]
        if 'standout' in attrs and undo:
            attrs.remove('standout')
            attrspec = attrspec.copy_modified(fg=','.join(attrs))
        elif 'standout' not in attrs and (not undo):
            attrs.append('standout')
            attrspec = attrspec.copy_modified(fg=','.join(attrs))
        return attrspec

    def reverse_video(self, undo: bool=False) -> None:
        """
        Reverse video/scanmode (DECSCNM) by swapping fg and bg colors.
        """
        for y in range(self.height):
            for x in range(self.width):
                char = self.term[y][x]
                attrs = self.reverse_attrspec(char[0], undo=undo)
                self.term[y][x] = (attrs,) + char[1:]

    def set_mode(self, mode: Literal[1, 3, 4, 5, 6, 7, 20, 25, 2004], flag: bool, qmark: bool, reset: bool) -> None:
        """
        Helper method for csi_set_modes: set single mode.
        """
        if qmark:
            if mode == 1:
                self.modes.keys_decckm = flag
            elif mode == 3:
                self.clear()
            elif mode == 5:
                if self.modes.reverse_video != flag:
                    self.reverse_video(undo=not flag)
                self.modes.reverse_video = flag
            elif mode == 6:
                self.modes.constrain_scrolling = flag
                self.set_term_cursor(0, 0)
            elif mode == 7:
                self.modes.autowrap = flag
            elif mode == 25:
                self.modes.visible_cursor = flag
                self.set_term_cursor()
            elif mode == 2004:
                self.modes.bracketed_paste = flag
        elif mode == 3:
            self.modes.display_ctrl = flag
        elif mode == 4:
            self.modes.insert = flag
        elif mode == 20:
            self.modes.lfnl = flag

    def csi_set_modes(self, modes: Iterable[int], qmark: bool, reset: bool=False) -> None:
        """
        Set (DECSET/ECMA-48) or reset modes (DECRST/ECMA-48) if reset is True.
        """
        flag = not reset
        for mode in modes:
            self.set_mode(mode, flag, qmark, reset)

    def csi_set_scroll(self, top: int=0, bottom: int=0) -> None:
        """
        Set scrolling region, 'top' is the line number of first line in the
        scrolling region. 'bottom' is the line number of bottom line. If both
        are set to 0, the whole screen will be used (default).
        """
        if not top:
            top = 1
        if not bottom:
            bottom = self.height
        if top < bottom <= self.height:
            self.scrollregion_start = self.constrain_coords(0, top - 1, ignore_scrolling=True)[1]
            self.scrollregion_end = self.constrain_coords(0, bottom - 1, ignore_scrolling=True)[1]
            self.set_term_cursor(0, 0)

    def csi_clear_tabstop(self, mode: Literal[0, 3]=0):
        """
        Clear tabstop at current position or if 'mode' is 3, delete all
        tabstops.
        """
        if mode == 0:
            self.set_tabstop(remove=True)
        elif mode == 3:
            self.set_tabstop(clear=True)

    def csi_get_device_attributes(self, qmark: bool) -> None:
        """
        Report device attributes (what are you?). In our case, we'll report
        ourself as a VT102 terminal.
        """
        if not qmark:
            self.widget.respond(f'{ESC}[?6c')

    def csi_status_report(self, mode: Literal[5, 6]) -> None:
        """
        Report various information about the terminal status.
        Information is queried by 'mode', where possible values are:
            5 -> device status report
            6 -> cursor position report
        """
        if mode == 5:
            self.widget.respond(f'{ESC}[0n')
        elif mode == 6:
            x, y = self.term_cursor
            self.widget.respond(ESC + f'[{y + 1:d};{x + 1:d}R')

    def csi_erase_line(self, mode: Literal[0, 1, 2]) -> None:
        """
        Erase current line, modes are:
            0 -> erase from cursor to end of line.
            1 -> erase from start of line to cursor.
            2 -> erase whole line.
        """
        x, y = self.term_cursor
        if mode == 0:
            self.erase(self.term_cursor, (self.width - 1, y))
        elif mode == 1:
            self.erase((0, y), (x, y))
        elif mode == 2:
            self.blank_line(y)

    def csi_erase_display(self, mode: Literal[0, 1, 2]) -> None:
        """
        Erase display, modes are:
            0 -> erase from cursor to end of display.
            1 -> erase from start to cursor.
            2 -> erase the whole display.
        """
        if mode == 0:
            self.erase(self.term_cursor, (self.width - 1, self.height - 1))
        if mode == 1:
            self.erase((0, 0), (self.term_cursor[0] - 1, self.term_cursor[1]))
        elif mode == 2:
            self.clear(cursor=self.term_cursor)

    def csi_set_keyboard_leds(self, mode: Literal[0, 1, 2, 3]=0) -> None:
        """
        Set keyboard LEDs, modes are:
            0 -> clear all LEDs
            1 -> set scroll lock LED
            2 -> set num lock LED
            3 -> set caps lock LED

        This currently just emits a signal, so it can be processed by another
        widget or the main application.
        """
        states = {0: 'clear', 1: 'scroll_lock', 2: 'num_lock', 3: 'caps_lock'}
        if mode in states:
            self.widget.leds(states[mode])

    def clear(self, cursor: tuple[int, int] | None=None) -> None:
        """
        Clears the whole terminal screen and resets the cursor position
        to (0, 0) or to the coordinates given by 'cursor'.
        """
        self.term = [self.empty_line() for _ in range(self.height)]
        if cursor is None:
            self.set_term_cursor(0, 0)
        else:
            self.set_term_cursor(*cursor)

    def cols(self) -> int:
        return self.width

    def rows(self) -> int:
        return self.height

    def content(self, trim_left: int=0, trim_top: int=0, cols: int | None=None, rows: int | None=None, attr=None) -> Iterable[list[tuple[object, Literal['0', 'U'] | None, bytes]]]:
        if self.scrolling_up == 0:
            yield from self.term
        else:
            buf = self.scrollback_buffer + self.term
            yield from buf[-(self.height + self.scrolling_up):-self.scrolling_up]

    def content_delta(self, other: Canvas):
        if other is self:
            return [self.cols()] * self.rows()
        return self.content()