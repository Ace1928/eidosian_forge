from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
class ColorLookupTable(object):
    """
    Inspired by pygments/formatters/terminal256.py
    """

    def __init__(self):
        self._win32_colors = self._build_color_table()
        self.best_match = {}

    @staticmethod
    def _build_color_table():
        """
        Build an RGB-to-256 color conversion table
        """
        FG = FOREGROUND_COLOR
        BG = BACKROUND_COLOR
        return [(0, 0, 0, FG.BLACK, BG.BLACK), (0, 0, 170, FG.BLUE, BG.BLUE), (0, 170, 0, FG.GREEN, BG.GREEN), (0, 170, 170, FG.CYAN, BG.CYAN), (170, 0, 0, FG.RED, BG.RED), (170, 0, 170, FG.MAGENTA, BG.MAGENTA), (170, 170, 0, FG.YELLOW, BG.YELLOW), (136, 136, 136, FG.GRAY, BG.GRAY), (68, 68, 255, FG.BLUE | FG.INTENSITY, BG.BLUE | BG.INTENSITY), (68, 255, 68, FG.GREEN | FG.INTENSITY, BG.GREEN | BG.INTENSITY), (68, 255, 255, FG.CYAN | FG.INTENSITY, BG.CYAN | BG.INTENSITY), (255, 68, 68, FG.RED | FG.INTENSITY, BG.RED | BG.INTENSITY), (255, 68, 255, FG.MAGENTA | FG.INTENSITY, BG.MAGENTA | BG.INTENSITY), (255, 255, 68, FG.YELLOW | FG.INTENSITY, BG.YELLOW | BG.INTENSITY), (68, 68, 68, FG.BLACK | FG.INTENSITY, BG.BLACK | BG.INTENSITY), (255, 255, 255, FG.GRAY | FG.INTENSITY, BG.GRAY | BG.INTENSITY)]

    def _closest_color(self, r, g, b):
        distance = 257 * 257 * 3
        fg_match = 0
        bg_match = 0
        for r_, g_, b_, fg_, bg_ in self._win32_colors:
            rd = r - r_
            gd = g - g_
            bd = b - b_
            d = rd * rd + gd * gd + bd * bd
            if d < distance:
                fg_match = fg_
                bg_match = bg_
                distance = d
        return (fg_match, bg_match)

    def _color_indexes(self, color):
        indexes = self.best_match.get(color, None)
        if indexes is None:
            try:
                rgb = int(str(color), 16)
            except ValueError:
                rgb = 0
            r = rgb >> 16 & 255
            g = rgb >> 8 & 255
            b = rgb & 255
            indexes = self._closest_color(r, g, b)
            self.best_match[color] = indexes
        return indexes

    def lookup_fg_color(self, fg_color):
        """
        Return the color for use in the
        `windll.kernel32.SetConsoleTextAttribute` API call.

        :param fg_color: Foreground as text. E.g. 'ffffff' or 'red'
        """
        if fg_color in FG_ANSI_COLORS:
            return FG_ANSI_COLORS[fg_color]
        else:
            return self._color_indexes(fg_color)[0]

    def lookup_bg_color(self, bg_color):
        """
        Return the color for use in the
        `windll.kernel32.SetConsoleTextAttribute` API call.

        :param bg_color: Background as text. E.g. 'ffffff' or 'red'
        """
        if bg_color in BG_ANSI_COLORS:
            return BG_ANSI_COLORS[bg_color]
        else:
            return self._color_indexes(bg_color)[1]