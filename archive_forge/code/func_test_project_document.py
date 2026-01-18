import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_project_document(project_dir):
    readme_path = project_dir / 'README.md'
    assert not readme_path.exists(), 'README already exists'
    result = CliRunner().invoke(app, ['project', 'document', str(project_dir), '-o', str(readme_path)])
    assert result.exit_code == 0
    assert readme_path.is_file()
    text = readme_path.read_text('utf-8')
    assert SAMPLE_PROJECT['description'] in text