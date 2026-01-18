from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def tree_entries_from_data(data: bytes) -> List[EntryTup]:
    """Reads the binary representation of a tree and returns tuples of Tree items

    :param data: data block with tree data (as bytes)
    :return: list(tuple(binsha, mode, tree_relative_path), ...)"""
    ord_zero = ord('0')
    space_ord = ord(' ')
    len_data = len(data)
    i = 0
    out = []
    while i < len_data:
        mode = 0
        while data[i] != space_ord:
            mode = (mode << 3) + (data[i] - ord_zero)
            i += 1
        i += 1
        ns = i
        while data[i] != 0:
            i += 1
        name_bytes = data[ns:i]
        name = safe_decode(name_bytes)
        i += 1
        sha = data[i:i + 20]
        i = i + 20
        out.append((sha, mode, name))
    return out