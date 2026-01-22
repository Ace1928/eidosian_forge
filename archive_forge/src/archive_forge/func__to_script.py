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
def _to_script(self, migration_script: MigrationScript) -> Optional[Script]:
    template_args: Dict[str, Any] = self.template_args.copy()
    if getattr(migration_script, '_needs_render', False):
        autogen_context = self._last_autogen_context
        autogen_context.imports = set()
        if migration_script.imports:
            autogen_context.imports.update(migration_script.imports)
        render._render_python_into_templatevars(autogen_context, migration_script, template_args)
    assert migration_script.rev_id is not None
    return self.script_directory.generate_revision(migration_script.rev_id, migration_script.message, refresh=True, head=migration_script.head, splice=migration_script.splice, branch_labels=migration_script.branch_label, version_path=migration_script.version_path, depends_on=migration_script.depends_on, **template_args)