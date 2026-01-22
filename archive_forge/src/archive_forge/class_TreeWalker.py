from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
class TreeWalker(ListWalker):
    """ListWalker-compatible class for displaying TreeWidgets

    positions are TreeNodes."""

    def __init__(self, start_from) -> None:
        """start_from: TreeNode with the initial focus."""
        self.focus = start_from

    def get_focus(self):
        widget = self.focus.get_widget()
        return (widget, self.focus)

    def set_focus(self, focus) -> None:
        self.focus = focus
        self._modified()

    def get_next(self, start_from) -> tuple[TreeWidget, TreeNode] | tuple[None, None]:
        target = start_from.get_widget().next_inorder()
        if target is None:
            return (None, None)
        return (target, target.get_node())

    def get_prev(self, start_from) -> tuple[TreeWidget, TreeNode] | tuple[None, None]:
        target = start_from.get_widget().prev_inorder()
        if target is None:
            return (None, None)
        return (target, target.get_node())