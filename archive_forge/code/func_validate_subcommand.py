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
def validate_subcommand(commands: Sequence[str], workflows: Sequence[str], subcommand: str) -> None:
    """Check that a subcommand is valid and defined. Raises an error otherwise.

    commands (Sequence[str]): The available commands.
    subcommand (str): The subcommand.
    """
    if not commands and (not workflows):
        msg.fail(f'No commands or workflows defined in {PROJECT_FILE}', exits=1)
    if subcommand not in commands and subcommand not in workflows:
        help_msg = []
        if subcommand in ['assets', 'asset']:
            help_msg.append('Did you mean to run: python -m weasel assets?')
        if commands:
            help_msg.append(f'Available commands: {', '.join(commands)}')
        if workflows:
            help_msg.append(f'Available workflows: {', '.join(workflows)}')
        msg.fail(f"Can't find command or workflow '{subcommand}' in {PROJECT_FILE}", '. '.join(help_msg), exits=1)