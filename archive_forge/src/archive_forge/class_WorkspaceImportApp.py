from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any
from jupyter_core.application import JupyterApp
from traitlets import Bool, Unicode
from ._version import __version__
from .config import LabConfig
from .workspaces_handler import WorkspacesManager
class WorkspaceImportApp(JupyterApp, LabConfig):
    """A workspace import app."""
    version = __version__
    description = '\n    Import a JupyterLab workspace\n\n    This command will import a workspace from a JSON file. The format of the\n        file must be the same as what the export functionality emits.\n    '
    workspace_name = Unicode(None, config=True, allow_none=True, help='\n        Workspace name. If given, the workspace ID in the imported\n        file will be replaced with a new ID pointing to this\n        workspace name.\n        ')
    aliases = {'name': 'WorkspaceImportApp.workspace_name'}

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the app."""
        super().initialize(*args, **kwargs)
        self.manager = WorkspacesManager(self.workspaces_dir)

    def start(self) -> None:
        """Start the app."""
        if len(self.extra_args) != 1:
            self.log.info('One argument is required for workspace import.')
            self.exit(1)
        with self._smart_open() as fid:
            try:
                workspace = self._validate(fid)
            except Exception as e:
                self.log.info('%s is not a valid workspace:\n%s', fid.name, e)
                self.exit(1)
        try:
            workspace_path = self.manager.save(workspace['metadata']['id'], json.dumps(workspace))
        except Exception as e:
            self.log.info('Workspace could not be exported:\n%s', e)
            self.exit(1)
        self.log.info('Saved workspace: %s', workspace_path)

    def _smart_open(self) -> Any:
        file_name = self.extra_args[0]
        if file_name == '-':
            return sys.stdin
        file_path = Path(file_name).resolve()
        if not file_path.exists():
            self.log.info('%s does not exist.', file_name)
            self.exit(1)
        return file_path.open(encoding='utf-8')

    def _validate(self, data: Any) -> Any:
        workspace = json.load(data)
        if 'data' not in workspace:
            msg = 'The `data` field is missing.'
            raise Exception(msg)
        if self.workspace_name is not None and self.workspace_name:
            workspace['metadata'] = {'id': self.workspace_name}
        elif 'id' not in workspace['metadata']:
            msg = 'The `id` field is missing in `metadata`.'
            raise Exception(msg)
        return workspace