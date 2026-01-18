import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def write_info_refs(refs, store: ObjectContainer):
    """Generate info refs."""
    from .object_store import peel_sha
    for name, sha in sorted(refs.items()):
        if name == HEADREF:
            continue
        try:
            o = store[sha]
        except KeyError:
            continue
        unpeeled, peeled = peel_sha(store, sha)
        yield (o.id + b'\t' + name + b'\n')
        if o.id != peeled.id:
            yield (peeled.id + b'\t' + name + PEELED_TAG_SUFFIX + b'\n')