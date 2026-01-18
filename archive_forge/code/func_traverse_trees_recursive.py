from stat import S_ISDIR
from git.compat import safe_decode, defenc
from typing import (
def traverse_trees_recursive(odb: 'GitCmdObjectDB', tree_shas: Sequence[Union[bytes, None]], path_prefix: str) -> List[Tuple[EntryTupOrNone, ...]]:
    """
    :return: list of list with entries according to the given binary tree-shas.
        The result is encoded in a list
        of n tuple|None per blob/commit, (n == len(tree_shas)), where:

        * [0] == 20 byte sha
        * [1] == mode as int
        * [2] == path relative to working tree root

        The entry tuple is None if the respective blob/commit did not
        exist in the given tree.

    :param tree_shas: iterable of shas pointing to trees. All trees must
        be on the same level. A tree-sha may be None in which case None.

    :param path_prefix: a prefix to be added to the returned paths on this level,
        set it '' for the first iteration.

    :note: The ordering of the returned items will be partially lost.
    """
    trees_data: List[List[EntryTupOrNone]] = []
    nt = len(tree_shas)
    for tree_sha in tree_shas:
        if tree_sha is None:
            data: List[EntryTupOrNone] = []
        else:
            data = list(tree_entries_from_data(odb.stream(tree_sha).read()))
        trees_data.append(data)
    out: List[Tuple[EntryTupOrNone, ...]] = []
    for ti, tree_data in enumerate(trees_data):
        for ii, item in enumerate(tree_data):
            if not item:
                continue
            entries: List[EntryTupOrNone]
            entries = [None for _ in range(nt)]
            entries[ti] = item
            _sha, mode, name = item
            is_dir = S_ISDIR(mode)
            for tio in range(ti + 1, ti + nt):
                tio = tio % nt
                entries[tio] = _find_by_name(trees_data[tio], name, is_dir, ii)
            if is_dir:
                out.extend(traverse_trees_recursive(odb, [ei and ei[0] or None for ei in entries], path_prefix + name + '/'))
            else:
                out.append(tuple((_to_full_path(e, path_prefix) for e in entries)))
            tree_data[ii] = None
        del tree_data[:]
    return out