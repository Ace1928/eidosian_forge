from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
class ParentNode(TreeNode):
    """Maintain sort order for TreeNodes."""

    def __init__(self, value: typing.Any, parent: ParentNode | None=None, key: Hashable=None, depth: int | None=None) -> None:
        super().__init__(value, parent=parent, key=key, depth=depth)
        self._child_keys: Sequence[Hashable] | None = None
        self._children: dict[Hashable, TreeNode] = {}

    def get_child_keys(self, reload: bool=False) -> Sequence[Hashable]:
        """Return a possibly ordered list of child keys"""
        if self._child_keys is None or reload:
            self._child_keys = self.load_child_keys()
        return self._child_keys

    def load_child_keys(self) -> Sequence[Hashable]:
        """Provide ParentNode with an ordered list of child keys (virtual function)"""
        raise TreeWidgetError('virtual function.  Implement in subclass')

    def get_child_widget(self, key) -> TreeWidget:
        """Return the widget for a given key.  Create if necessary."""
        return self.get_child_node(key).get_widget()

    def get_child_node(self, key, reload: bool=False) -> TreeNode:
        """Return the child node for a given key. Create if necessary."""
        if key not in self._children or reload:
            self._children[key] = self.load_child_node(key)
        return self._children[key]

    def load_child_node(self, key: Hashable) -> TreeNode:
        """Load the child node for a given key (virtual function)"""
        raise TreeWidgetError('virtual function.  Implement in subclass')

    def set_child_node(self, key: Hashable, node: TreeNode) -> None:
        """Set the child node for a given key.

        Useful for bottom-up, lazy population of a tree.
        """
        self._children[key] = node

    def change_child_key(self, oldkey: Hashable, newkey: Hashable) -> None:
        if newkey in self._children:
            raise TreeWidgetError(f'{newkey} is already in use')
        self._children[newkey] = self._children.pop(oldkey)
        self._children[newkey].set_key(newkey)

    def get_child_index(self, key: Hashable) -> int:
        try:
            return self.get_child_keys().index(key)
        except ValueError as exc:
            raise TreeWidgetError(f"Can't find key {key} in ParentNode {self.get_key()}\nParentNode items: {self.get_child_keys()!s}").with_traceback(exc.__traceback__) from exc

    def next_child(self, key: Hashable) -> TreeNode | None:
        """Return the next child node in index order from the given key."""
        index = self.get_child_index(key)
        if index is None:
            return None
        index += 1
        child_keys = self.get_child_keys()
        if index < len(child_keys):
            return self.get_child_node(child_keys[index])
        return None

    def prev_child(self, key: Hashable) -> TreeNode | None:
        """Return the previous child node in index order from the given key."""
        index = self.get_child_index(key)
        if index is None:
            return None
        child_keys = self.get_child_keys()
        index -= 1
        if index >= 0:
            return self.get_child_node(child_keys[index])
        return None

    def get_first_child(self) -> TreeNode:
        """Return the first TreeNode in the directory."""
        child_keys = self.get_child_keys()
        return self.get_child_node(child_keys[0])

    def get_last_child(self) -> TreeNode:
        """Return the last TreeNode in the directory."""
        child_keys = self.get_child_keys()
        return self.get_child_node(child_keys[-1])

    def has_children(self) -> bool:
        """Does this node have any children?"""
        return len(self.get_child_keys()) > 0