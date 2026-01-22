import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def add_packed_refs(self, new_refs: Dict[Ref, Optional[ObjectID]]):
    """Add the given refs as packed refs.

        Args:
          new_refs: A mapping of ref names to targets; if a target is None that
            means remove the ref
        """
    if not new_refs:
        return
    path = os.path.join(self.path, b'packed-refs')
    with GitFile(path, 'wb') as f:
        packed_refs = self.get_packed_refs().copy()
        for ref, target in new_refs.items():
            if ref == HEADREF:
                raise ValueError('cannot pack HEAD')
            with suppress(OSError):
                os.remove(self.refpath(ref))
            if target is not None:
                packed_refs[ref] = target
            else:
                packed_refs.pop(ref, None)
        write_packed_refs(f, packed_refs, self._peeled_refs)
        self._packed_refs = packed_refs