import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def _import_remote_refs(refs_container: RefsContainer, remote_name: str, refs: Dict[str, str], message: Optional[bytes]=None, prune: bool=False, prune_tags: bool=False):
    stripped_refs = strip_peeled_refs(refs)
    branches = {n[len(LOCAL_BRANCH_PREFIX):]: v for n, v in stripped_refs.items() if n.startswith(LOCAL_BRANCH_PREFIX)}
    refs_container.import_refs(b'refs/remotes/' + remote_name.encode(), branches, message=message, prune=prune)
    tags = {n[len(LOCAL_TAG_PREFIX):]: v for n, v in stripped_refs.items() if n.startswith(LOCAL_TAG_PREFIX) and (not n.endswith(PEELED_TAG_SUFFIX))}
    refs_container.import_refs(LOCAL_TAG_PREFIX, tags, message=message, prune=prune_tags)