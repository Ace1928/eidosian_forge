import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _find_exact_renames(self):
    add_map = defaultdict(list)
    for add in self._adds:
        add_map[add.new.sha].append(add.new)
    delete_map = defaultdict(list)
    for delete in self._deletes:
        is_delete = delete.type == CHANGE_DELETE
        delete_map[delete.old.sha].append((delete.old, is_delete))
    add_paths = set()
    delete_paths = set()
    for sha, sha_deletes in delete_map.items():
        sha_adds = add_map[sha]
        for (old, is_delete), new in zip(sha_deletes, sha_adds):
            if stat.S_IFMT(old.mode) != stat.S_IFMT(new.mode):
                continue
            if is_delete:
                delete_paths.add(old.path)
            add_paths.add(new.path)
            new_type = is_delete and CHANGE_RENAME or CHANGE_COPY
            self._changes.append(TreeChange(new_type, old, new))
        num_extra_adds = len(sha_adds) - len(sha_deletes)
        old = sha_deletes[0][0]
        if num_extra_adds > 0:
            for new in sha_adds[-num_extra_adds:]:
                add_paths.add(new.path)
                self._changes.append(TreeChange(CHANGE_COPY, old, new))
    self._prune(add_paths, delete_paths)