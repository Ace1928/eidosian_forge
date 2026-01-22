from __future__ import annotations
from contextlib import contextmanager
import datetime
import os
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
@classmethod
def _from_filename(cls, scriptdir: ScriptDirectory, dir_: str, filename: str) -> Optional[Script]:
    if scriptdir.sourceless:
        py_match = _sourceless_rev_file.match(filename)
    else:
        py_match = _only_source_rev_file.match(filename)
    if not py_match:
        return None
    py_filename = py_match.group(1)
    if scriptdir.sourceless:
        is_c = py_match.group(2) == 'c'
        is_o = py_match.group(2) == 'o'
    else:
        is_c = is_o = False
    if is_o or is_c:
        py_exists = os.path.exists(os.path.join(dir_, py_filename))
        pyc_exists = os.path.exists(os.path.join(dir_, py_filename + 'c'))
        if py_exists or (is_o and pyc_exists):
            return None
    module = util.load_python_file(dir_, filename)
    if not hasattr(module, 'revision'):
        m = _legacy_rev.match(filename)
        if not m:
            raise util.CommandError("Could not determine revision id from filename %s. Be sure the 'revision' variable is declared inside the script (please see 'Upgrading from Alembic 0.1 to 0.2' in the documentation)." % filename)
        else:
            revision = m.group(1)
    else:
        revision = module.revision
    return Script(module, revision, os.path.join(dir_, filename))