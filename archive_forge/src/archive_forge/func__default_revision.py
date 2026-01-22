from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
def _default_revision(self) -> MigrationScript:
    command_args: Dict[str, Any] = self.command_args
    op = ops.MigrationScript(rev_id=command_args['rev_id'] or util.rev_id(), message=command_args['message'], upgrade_ops=ops.UpgradeOps([]), downgrade_ops=ops.DowngradeOps([]), head=command_args['head'], splice=command_args['splice'], branch_label=command_args['branch_label'], version_path=command_args['version_path'], depends_on=command_args['depends_on'])
    return op