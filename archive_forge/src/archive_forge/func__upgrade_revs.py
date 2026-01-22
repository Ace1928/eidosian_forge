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
def _upgrade_revs(self, destination: str, current_rev: str) -> List[RevisionStep]:
    with self._catch_revision_errors(ancestor='Destination %(end)s is not a valid upgrade target from current head(s)', end=destination):
        revs = self.iterate_revisions(destination, current_rev, implicit_base=True)
        return [migration.MigrationStep.upgrade_from_script(self.revision_map, script) for script in reversed(list(revs))]