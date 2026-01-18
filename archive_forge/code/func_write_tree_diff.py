import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def write_tree_diff(f, store, old_tree, new_tree, diff_binary=False):
    """Write tree diff.

    Args:
      f: File-like object to write to.
      old_tree: Old tree id
      new_tree: New tree id
      diff_binary: Whether to diff files even if they
        are considered binary files by is_binary().
    """
    changes = store.tree_changes(old_tree, new_tree)
    for (oldpath, newpath), (oldmode, newmode), (oldsha, newsha) in changes:
        write_object_diff(f, store, (oldpath, oldmode, oldsha), (newpath, newmode, newsha), diff_binary=diff_binary)