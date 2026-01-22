from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
class BaseScreen(metaclass=BaseMeta):
    """
    Base class for Screen classes (raw_display.Screen, .. etc)
    """
    signals: typing.ClassVar[list[str]] = [UPDATE_PALETTE_ENTRY, INPUT_DESCRIPTORS_CHANGED]

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'{self.__class__.__module__}.{self.__class__.__name__}')
        self._palette: dict[str | None, tuple[AttrSpec, AttrSpec, AttrSpec, AttrSpec, AttrSpec]] = {}
        self._started: bool = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self, *args, **kwargs) -> StoppingContext:
        """Set up the screen.  If the screen has already been started, does
        nothing.

        May be used as a context manager, in which case :meth:`stop` will
        automatically be called at the end of the block:

            with screen.start():
                ...

        You shouldn't override this method in a subclass; instead, override
        :meth:`_start`.
        """
        if not self._started:
            self._started = True
            self._start(*args, **kwargs)
        return StoppingContext(self)

    def _start(self) -> None:
        pass

    def stop(self) -> None:
        if self._started:
            self._stop()
        self._started = False

    def _stop(self) -> None:
        pass

    def run_wrapper(self, fn, *args, **kwargs):
        """Start the screen, call a function, then stop the screen.  Extra
        arguments are passed to `start`.

        Deprecated in favor of calling `start` as a context manager.
        """
        warnings.warn('run_wrapper is deprecated in favor of calling `start` as a context manager.', DeprecationWarning, stacklevel=3)
        with self.start(*args, **kwargs):
            return fn()

    def set_mouse_tracking(self, enable: bool=True) -> None:
        pass

    @abc.abstractmethod
    def draw_screen(self, size: tuple[int, int], canvas: Canvas) -> None:
        pass

    def clear(self) -> None:
        """Clear the screen if possible.

        Force the screen to be completely repainted on the next call to draw_screen().
        """

    def get_cols_rows(self) -> tuple[int, int]:
        """Return the terminal dimensions (num columns, num rows).

        Default (fallback) is 80x24.
        """
        return (80, 24)

    def register_palette(self, palette: Iterable[tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, str] | tuple[str, str, str, str, str, str]]) -> None:
        """Register a set of palette entries.

        palette -- a list of (name, like_other_name) or
        (name, foreground, background, mono, foreground_high, background_high) tuples

            The (name, like_other_name) format will copy the settings
            from the palette entry like_other_name, which must appear
            before this tuple in the list.

            The mono and foreground/background_high values are
            optional ie. the second tuple format may have 3, 4 or 6
            values.  See register_palette_entry() for a description
            of the tuple values.
        """
        for item in palette:
            if len(item) in {3, 4, 6}:
                self.register_palette_entry(*item)
                continue
            if len(item) != 2:
                raise ScreenError(f'Invalid register_palette entry: {item!r}')
            name, like_name = item
            if like_name not in self._palette:
                raise ScreenError(f"palette entry '{like_name}' doesn't exist")
            self._palette[name] = self._palette[like_name]

    def register_palette_entry(self, name: str | None, foreground: str, background: str, mono: str | None=None, foreground_high: str | None=None, background_high: str | None=None) -> None:
        """Register a single palette entry.

        name -- new entry/attribute name

        foreground -- a string containing a comma-separated foreground
        color and settings

            Color values:
            'default' (use the terminal's default foreground),
            'black', 'dark red', 'dark green', 'brown', 'dark blue',
            'dark magenta', 'dark cyan', 'light gray', 'dark gray',
            'light red', 'light green', 'yellow', 'light blue',
            'light magenta', 'light cyan', 'white'

            Settings:
            'bold', 'underline', 'blink', 'standout', 'strikethrough'

            Some terminals use 'bold' for bright colors.  Most terminals
            ignore the 'blink' setting.  If the color is not given then
            'default' will be assumed.

        background -- a string containing the background color

            Background color values:
            'default' (use the terminal's default background),
            'black', 'dark red', 'dark green', 'brown', 'dark blue',
            'dark magenta', 'dark cyan', 'light gray'

        mono -- a comma-separated string containing monochrome terminal
        settings (see "Settings" above.)

            None = no terminal settings (same as 'default')

        foreground_high -- a string containing a comma-separated
        foreground color and settings, standard foreground
        colors (see "Color values" above) or high-colors may
        be used

            High-color example values:
            '#009' (0% red, 0% green, 60% red, like HTML colors)
            '#fcc' (100% red, 80% green, 80% blue)
            'g40' (40% gray, decimal), 'g#cc' (80% gray, hex),
            '#000', 'g0', 'g#00' (black),
            '#fff', 'g100', 'g#ff' (white)
            'h8' (color number 8), 'h255' (color number 255)

            None = use foreground parameter value

        background_high -- a string containing the background color,
        standard background colors (see "Background colors" above)
        or high-colors (see "High-color example values" above)
        may be used

            None = use background parameter value
        """
        basic = AttrSpec(foreground, background, 16)
        if isinstance(mono, tuple):
            mono = ','.join(mono)
        if mono is None:
            mono = DEFAULT
        mono_spec = AttrSpec(mono, DEFAULT, 1)
        if foreground_high is None:
            foreground_high = foreground
        if background_high is None:
            background_high = background
        high_256 = AttrSpec(foreground_high, background_high, 256)
        high_true = AttrSpec(foreground_high, background_high, 2 ** 24)

        def large_h(desc: str) -> bool:
            if not desc.startswith('h'):
                return False
            if ',' in desc:
                desc = desc.split(',', 1)[0]
            num = int(desc[1:], 10)
            return num > 15
        if large_h(foreground_high) or large_h(background_high):
            high_88 = basic
        else:
            high_88 = AttrSpec(foreground_high, background_high, 88)
        signals.emit_signal(self, UPDATE_PALETTE_ENTRY, name, basic, mono_spec, high_88, high_256, high_true)
        self._palette[name] = (basic, mono_spec, high_88, high_256, high_true)