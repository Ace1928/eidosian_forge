from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
class Win32Output(Output):
    """
    I/O abstraction for rendering to Windows consoles.
    (cmd.exe and similar.)
    """

    def __init__(self, stdout, use_complete_width=False):
        self.use_complete_width = use_complete_width
        self._buffer = []
        self.stdout = stdout
        self.hconsole = windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        self._in_alternate_screen = False
        self.color_lookup_table = ColorLookupTable()
        info = self.get_win32_screen_buffer_info()
        self.default_attrs = info.wAttributes if info else 15
        if _DEBUG_RENDER_OUTPUT:
            self.LOG = open(_DEBUG_RENDER_OUTPUT_FILENAME, 'ab')

    def fileno(self):
        """ Return file descriptor. """
        return self.stdout.fileno()

    def encoding(self):
        """ Return encoding used for stdout. """
        return self.stdout.encoding

    def write(self, data):
        self._buffer.append(data)

    def write_raw(self, data):
        """ For win32, there is no difference between write and write_raw. """
        self.write(data)

    def get_size(self):
        from prompt_toolkit.layout.screen import Size
        info = self.get_win32_screen_buffer_info()
        if self.use_complete_width:
            width = info.dwSize.X
        else:
            width = info.srWindow.Right - info.srWindow.Left
        height = info.srWindow.Bottom - info.srWindow.Top + 1
        maxwidth = info.dwSize.X - 1
        width = min(maxwidth, width)
        return Size(rows=height, columns=width)

    def _winapi(self, func, *a, **kw):
        """
        Flush and call win API function.
        """
        self.flush()
        if _DEBUG_RENDER_OUTPUT:
            self.LOG.write(('%r' % func.__name__).encode('utf-8') + b'\n')
            self.LOG.write(b'     ' + ', '.join(['%r' % i for i in a]).encode('utf-8') + b'\n')
            self.LOG.write(b'     ' + ', '.join(['%r' % type(i) for i in a]).encode('utf-8') + b'\n')
            self.LOG.flush()
        try:
            return func(*a, **kw)
        except ArgumentError as e:
            if _DEBUG_RENDER_OUTPUT:
                self.LOG.write(('    Error in %r %r %s\n' % (func.__name__, e, e)).encode('utf-8'))

    def get_win32_screen_buffer_info(self):
        """
        Return Screen buffer info.
        """
        self.flush()
        sbinfo = CONSOLE_SCREEN_BUFFER_INFO()
        success = windll.kernel32.GetConsoleScreenBufferInfo(self.hconsole, byref(sbinfo))
        if success:
            return sbinfo
        else:
            raise NoConsoleScreenBufferError

    def set_title(self, title):
        """
        Set terminal title.
        """
        assert isinstance(title, six.text_type)
        self._winapi(windll.kernel32.SetConsoleTitleW, title)

    def clear_title(self):
        self._winapi(windll.kernel32.SetConsoleTitleW, '')

    def erase_screen(self):
        start = COORD(0, 0)
        sbinfo = self.get_win32_screen_buffer_info()
        length = sbinfo.dwSize.X * sbinfo.dwSize.Y
        self.cursor_goto(row=0, column=0)
        self._erase(start, length)

    def erase_down(self):
        sbinfo = self.get_win32_screen_buffer_info()
        size = sbinfo.dwSize
        start = sbinfo.dwCursorPosition
        length = size.X - size.X + size.X * (size.Y - sbinfo.dwCursorPosition.Y)
        self._erase(start, length)

    def erase_end_of_line(self):
        """
        """
        sbinfo = self.get_win32_screen_buffer_info()
        start = sbinfo.dwCursorPosition
        length = sbinfo.dwSize.X - sbinfo.dwCursorPosition.X
        self._erase(start, length)

    def _erase(self, start, length):
        chars_written = c_ulong()
        self._winapi(windll.kernel32.FillConsoleOutputCharacterA, self.hconsole, c_char(b' '), DWORD(length), _coord_byval(start), byref(chars_written))
        sbinfo = self.get_win32_screen_buffer_info()
        self._winapi(windll.kernel32.FillConsoleOutputAttribute, self.hconsole, sbinfo.wAttributes, length, _coord_byval(start), byref(chars_written))

    def reset_attributes(self):
        """ Reset the console foreground/background color. """
        self._winapi(windll.kernel32.SetConsoleTextAttribute, self.hconsole, self.default_attrs)

    def set_attributes(self, attrs):
        fgcolor, bgcolor, bold, underline, italic, blink, reverse = attrs
        attrs = self.default_attrs
        if fgcolor is not None:
            attrs = attrs & ~15
            attrs |= self.color_lookup_table.lookup_fg_color(fgcolor)
        if bgcolor is not None:
            attrs = attrs & ~240
            attrs |= self.color_lookup_table.lookup_bg_color(bgcolor)
        if reverse:
            attrs = attrs & ~255 | (attrs & 15) << 4 | (attrs & 240) >> 4
        self._winapi(windll.kernel32.SetConsoleTextAttribute, self.hconsole, attrs)

    def disable_autowrap(self):
        pass

    def enable_autowrap(self):
        pass

    def cursor_goto(self, row=0, column=0):
        pos = COORD(x=column, y=row)
        self._winapi(windll.kernel32.SetConsoleCursorPosition, self.hconsole, _coord_byval(pos))

    def cursor_up(self, amount):
        sr = self.get_win32_screen_buffer_info().dwCursorPosition
        pos = COORD(sr.X, sr.Y - amount)
        self._winapi(windll.kernel32.SetConsoleCursorPosition, self.hconsole, _coord_byval(pos))

    def cursor_down(self, amount):
        self.cursor_up(-amount)

    def cursor_forward(self, amount):
        sr = self.get_win32_screen_buffer_info().dwCursorPosition
        pos = COORD(max(0, sr.X + amount), sr.Y)
        self._winapi(windll.kernel32.SetConsoleCursorPosition, self.hconsole, _coord_byval(pos))

    def cursor_backward(self, amount):
        self.cursor_forward(-amount)

    def flush(self):
        """
        Write to output stream and flush.
        """
        if not self._buffer:
            self.stdout.flush()
            return
        data = ''.join(self._buffer)
        if _DEBUG_RENDER_OUTPUT:
            self.LOG.write(('%r' % data).encode('utf-8') + b'\n')
            self.LOG.flush()
        for b in data:
            written = DWORD()
            retval = windll.kernel32.WriteConsoleW(self.hconsole, b, 1, byref(written), None)
            assert retval != 0
        self._buffer = []

    def get_rows_below_cursor_position(self):
        info = self.get_win32_screen_buffer_info()
        return info.srWindow.Bottom - info.dwCursorPosition.Y + 1

    def scroll_buffer_to_prompt(self):
        """
        To be called before drawing the prompt. This should scroll the console
        to left, with the cursor at the bottom (if possible).
        """
        info = self.get_win32_screen_buffer_info()
        sr = info.srWindow
        cursor_pos = info.dwCursorPosition
        result = SMALL_RECT()
        result.Left = 0
        result.Right = sr.Right - sr.Left
        win_height = sr.Bottom - sr.Top
        if 0 < sr.Bottom - cursor_pos.Y < win_height - 1:
            result.Bottom = sr.Bottom
        else:
            result.Bottom = max(win_height, cursor_pos.Y)
        result.Top = result.Bottom - win_height
        self._winapi(windll.kernel32.SetConsoleWindowInfo, self.hconsole, True, byref(result))

    def enter_alternate_screen(self):
        """
        Go to alternate screen buffer.
        """
        if not self._in_alternate_screen:
            GENERIC_READ = 2147483648
            GENERIC_WRITE = 1073741824
            handle = self._winapi(windll.kernel32.CreateConsoleScreenBuffer, GENERIC_READ | GENERIC_WRITE, DWORD(0), None, DWORD(1), None)
            self._winapi(windll.kernel32.SetConsoleActiveScreenBuffer, handle)
            self.hconsole = handle
            self._in_alternate_screen = True

    def quit_alternate_screen(self):
        """
        Make stdout again the active buffer.
        """
        if self._in_alternate_screen:
            stdout = self._winapi(windll.kernel32.GetStdHandle, STD_OUTPUT_HANDLE)
            self._winapi(windll.kernel32.SetConsoleActiveScreenBuffer, stdout)
            self._winapi(windll.kernel32.CloseHandle, self.hconsole)
            self.hconsole = stdout
            self._in_alternate_screen = False

    def enable_mouse_support(self):
        ENABLE_MOUSE_INPUT = 16
        handle = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        original_mode = DWORD()
        self._winapi(windll.kernel32.GetConsoleMode, handle, pointer(original_mode))
        self._winapi(windll.kernel32.SetConsoleMode, handle, original_mode.value | ENABLE_MOUSE_INPUT)

    def disable_mouse_support(self):
        ENABLE_MOUSE_INPUT = 16
        handle = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
        original_mode = DWORD()
        self._winapi(windll.kernel32.GetConsoleMode, handle, pointer(original_mode))
        self._winapi(windll.kernel32.SetConsoleMode, handle, original_mode.value & ~ENABLE_MOUSE_INPUT)

    def hide_cursor(self):
        pass

    def show_cursor(self):
        pass

    @classmethod
    def win32_refresh_window(cls):
        """
        Call win32 API to refresh the whole Window.

        This is sometimes necessary when the application paints background
        for completion menus. When the menu disappears, it leaves traces due
        to a bug in the Windows Console. Sending a repaint request solves it.
        """
        handle = windll.kernel32.GetConsoleWindow()
        RDW_INVALIDATE = 1
        windll.user32.RedrawWindow(handle, None, None, c_uint(RDW_INVALIDATE))