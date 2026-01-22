import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from wasabi import msg
from ..util import get_hash, join_command, load_project_config, run_command, working_dir
from .main import COMMAND, NAME, PROJECT_FILE, Arg, Opt, app
Ensure that the "dvc" command is available and that the current project
    directory is an initialized DVC project.
    