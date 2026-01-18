import pytest
from pathlib import Path
import ast
import tokenize
import numpy
@pytest.mark.slow
def test_warning_calls():
    base = Path(numpy.__file__).parent
    for path in base.rglob('*.py'):
        if base / 'testing' in path.parents:
            continue
        if path == base / '__init__.py':
            continue
        if path == base / 'random' / '__init__.py':
            continue
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read())
            FindFuncs(path).visit(tree)