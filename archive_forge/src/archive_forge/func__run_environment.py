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
def _run_environment(self, rev: _GetRevArg, migration_context: MigrationContext, autogenerate: bool) -> None:
    if autogenerate:
        if self.command_args['sql']:
            raise util.CommandError('Using --sql with --autogenerate does not make any sense')
        if set(self.script_directory.get_revisions(rev)) != set(self.script_directory.get_revisions('heads')):
            raise util.CommandError('Target database is not up to date.')
    upgrade_token = migration_context.opts['upgrade_token']
    downgrade_token = migration_context.opts['downgrade_token']
    migration_script = self.generated_revisions[-1]
    if not getattr(migration_script, '_needs_render', False):
        migration_script.upgrade_ops_list[-1].upgrade_token = upgrade_token
        migration_script.downgrade_ops_list[-1].downgrade_token = downgrade_token
        migration_script._needs_render = True
    else:
        migration_script._upgrade_ops.append(ops.UpgradeOps([], upgrade_token=upgrade_token))
        migration_script._downgrade_ops.append(ops.DowngradeOps([], downgrade_token=downgrade_token))
    autogen_context = AutogenContext(migration_context, autogenerate=autogenerate)
    self._last_autogen_context: AutogenContext = autogen_context
    if autogenerate:
        compare._populate_migration_script(autogen_context, migration_script)
    if self.process_revision_directives:
        self.process_revision_directives(migration_context, rev, self.generated_revisions)
    hook = migration_context.opts['process_revision_directives']
    if hook:
        hook(migration_context, rev, self.generated_revisions)
    for migration_script in self.generated_revisions:
        migration_script._needs_render = True