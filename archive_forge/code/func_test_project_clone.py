import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
@pytest.mark.skipif(not has_git(), reason='git not installed')
@pytest.mark.parametrize('options', ['', '--branch v3', '--repo https://github.com/explosion/projects --branch v3'])
def test_project_clone(options):
    with make_tempdir() as workspace:
        out = workspace / 'project'
        target = 'benchmarks/ner_conll03'
        if not options:
            options = []
        else:
            options = options.split()
        result = CliRunner().invoke(app, ['project', 'clone', target, *options, str(out)])
        assert result.exit_code == 0
        assert (out / 'README.md').is_file()