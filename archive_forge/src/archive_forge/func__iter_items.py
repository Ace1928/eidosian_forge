import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def _iter_items(cls: Type[T_References], repo: 'Repo', common_path: Union[PathLike, None]=None) -> Iterator[T_References]:
    if common_path is None:
        common_path = cls._common_path_default
    rela_paths = set()
    for root, dirs, files in os.walk(join_path_native(repo.common_dir, common_path)):
        if 'refs' not in root.split(os.sep):
            refs_id = [d for d in dirs if d == 'refs']
            if refs_id:
                dirs[0:] = ['refs']
        for f in files:
            if f == 'packed-refs':
                continue
            abs_path = to_native_path_linux(join_path(root, f))
            rela_paths.add(abs_path.replace(to_native_path_linux(repo.common_dir) + '/', ''))
    for _sha, rela_path in cls._iter_packed_refs(repo):
        if rela_path.startswith(str(common_path)):
            rela_paths.add(rela_path)
    for path in sorted(rela_paths):
        try:
            yield cls.from_path(repo, path)
        except ValueError:
            continue