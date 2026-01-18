import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_project_push_pull(project_dir):
    proj = dict(SAMPLE_PROJECT)
    remote = 'xyz'
    with make_tempdir() as remote_dir:
        proj['remotes'] = {remote: str(remote_dir)}
        proj_text = srsly.yaml_dumps(proj)
        (project_dir / 'project.yml').write_text(proj_text)
        test_file = project_dir / CFG_FILE
        result = CliRunner().invoke(app, ['project', 'run', 'create', str(project_dir)])
        assert result.exit_code == 0
        assert test_file.is_file()
        result = CliRunner().invoke(app, ['project', 'push', remote, str(project_dir)])
        assert result.exit_code == 0
        test_file.unlink()
        assert not test_file.exists()
        result = CliRunner().invoke(app, ['project', 'pull', remote, str(project_dir)])
        assert result.exit_code == 0
        assert test_file.is_file()