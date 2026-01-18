from __future__ import annotations
from .prettytable import PrettyTable
def update_theme(self) -> None:
    theme = self._theme
    self._vertical_char = theme.vertical_color + theme.vertical_char + RESET_CODE + theme.default_color
    self._horizontal_char = theme.horizontal_color + theme.horizontal_char + RESET_CODE + theme.default_color
    self._junction_char = theme.junction_color + theme.junction_char + RESET_CODE + theme.default_color