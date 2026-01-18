from pathlib import Path
from typing import Any, Dict
import pytest
import srsly
from typer.testing import CliRunner
from weasel import app
from weasel.cli.main import HELP
from weasel.util import get_git_version
Basic test to confirm help text appears