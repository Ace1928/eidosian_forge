from __future__ import annotations
import abc
import enum
import typing
import warnings
from .constants import Sizing, WHSettings
class WidgetContainerListContentsMixin:
    """
    Mixin class for widget containers whose positions are indexes into
    a list available as self.contents.
    """

    def __iter__(self) -> Iterator[int]:
        """
        Return an iterable of positions for this container from first
        to last.
        """
        return iter(range(len(self.contents)))

    def __reversed__(self) -> Iterator[int]:
        """
        Return an iterable of positions for this container from last
        to first.
        """
        return iter(range(len(self.contents) - 1, -1, -1))

    def __len__(self) -> int:
        return len(self.contents)

    @property
    @abc.abstractmethod
    def contents(self) -> list[tuple[Widget, typing.Any]]:
        """The contents of container as a list of (widget, options)"""

    @contents.setter
    def contents(self, new_contents: list[tuple[Widget, typing.Any]]) -> None:
        """The contents of container as a list of (widget, options)"""

    def _get_contents(self) -> list[tuple[Widget, typing.Any]]:
        warnings.warn(f'method `{self.__class__.__name__}._get_contents` is deprecated, please use `{self.__class__.__name__}.contents` property', DeprecationWarning, stacklevel=2)
        return self.contents

    def _set_contents(self, c: list[tuple[Widget, typing.Any]]) -> None:
        warnings.warn(f'method `{self.__class__.__name__}._set_contents` is deprecated, please use `{self.__class__.__name__}.contents` property', DeprecationWarning, stacklevel=2)
        self.contents = c

    @property
    @abc.abstractmethod
    def focus_position(self) -> int | None:
        """
        index of child widget in focus.
        """

    @focus_position.setter
    def focus_position(self, position: int) -> None:
        """
        index of child widget in focus.
        """

    def _get_focus_position(self) -> int | None:
        warnings.warn(f'method `{self.__class__.__name__}._get_focus_position` is deprecated, please use `{self.__class__.__name__}.focus_position` property', DeprecationWarning, stacklevel=3)
        return self.focus_position

    def _set_focus_position(self, position: int) -> None:
        """
        Set the widget in focus.

        position -- index of child widget to be made focus
        """
        warnings.warn(f'method `{self.__class__.__name__}._set_focus_position` is deprecated, please use `{self.__class__.__name__}.focus_position` property', DeprecationWarning, stacklevel=3)
        self.focus_position = position