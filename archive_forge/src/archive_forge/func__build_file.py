import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def _build_file(self, source: Path):
    """Build an artifact."""
    artifacts, command = self._get_artifacts(source)
    for artifact in artifacts:
        if opt_force or requires_rebuild([source], artifact):
            run_command(command, cwd=self.project.project_file.parent)
        self._build_file(artifact)