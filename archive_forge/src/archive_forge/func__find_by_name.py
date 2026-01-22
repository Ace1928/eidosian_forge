from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def _find_by_name(tree_data: MutableSequence[EntryTupOrNone], name: str, is_dir: bool, start_at: int) -> EntryTupOrNone:
    """Return data entry matching the given name and tree mode or None.

    Before the item is returned, the respective data item is set None in the
    tree_data list to mark it done.
    """
    try:
        item = tree_data[start_at]
        if item and item[2] == name and (S_ISDIR(item[1]) == is_dir):
            tree_data[start_at] = None
            return item
    except IndexError:
        pass
    for index, item in enumerate(tree_data):
        if item and item[2] == name and (S_ISDIR(item[1]) == is_dir):
            tree_data[index] = None
            return item
    return None