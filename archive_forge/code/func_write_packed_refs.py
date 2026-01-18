import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def write_packed_refs(f, packed_refs, peeled_refs=None):
    """Write a packed refs file.

    Args:
      f: empty file-like object to write to
      packed_refs: dict of refname to sha of packed refs to write
      peeled_refs: dict of refname to peeled value of sha
    """
    if peeled_refs is None:
        peeled_refs = {}
    else:
        f.write(b'# pack-refs with: peeled\n')
    for refname in sorted(packed_refs.keys()):
        f.write(git_line(packed_refs[refname], refname))
        if refname in peeled_refs:
            f.write(b'^' + peeled_refs[refname] + b'\n')