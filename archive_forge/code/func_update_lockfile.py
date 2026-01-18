import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import srsly
import typer
from wasabi import msg
from wasabi.util import locale_escape
from ..util import SimpleFrozenDict, SimpleFrozenList, check_spacy_env_vars
from ..util import get_checksum, get_hash, is_cwd, join_command, load_project_config
from ..util import parse_config_overrides, run_command, split_command, working_dir
from .main import COMMAND, PROJECT_FILE, PROJECT_LOCK, Arg, Opt, _get_parent_command
from .main import app
def update_lockfile(project_dir: Path, command: Dict[str, Any]) -> None:
    """Update the lockfile after running a command. Will create a lockfile if
    it doesn't yet exist and will add an entry for the current command, its
    script and dependencies/outputs.

    project_dir (Path): The current project directory.
    command (Dict[str, Any]): The command, as defined in the project.yml.
    """
    lock_path = project_dir / PROJECT_LOCK
    if not lock_path.exists():
        srsly.write_yaml(lock_path, {})
        data = {}
    else:
        data = srsly.read_yaml(lock_path)
    data[command['name']] = get_lock_entry(project_dir, command)
    srsly.write_yaml(lock_path, data)