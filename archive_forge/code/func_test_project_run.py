import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_project_run(project_dir):
    test_file = project_dir / CFG_FILE
    result = CliRunner().invoke(app, ['project', 'run', '--dry', 'create', str(project_dir)])
    assert result.exit_code == 0
    assert not test_file.is_file()
    result = CliRunner().invoke(app, ['project', 'run', 'create', str(project_dir)])
    assert result.exit_code == 0
    assert test_file.is_file()
    result = CliRunner().invoke(app, ['project', 'run', 'ok', str(project_dir)])
    assert result.exit_code == 0
    assert 'okokok' in result.stdout