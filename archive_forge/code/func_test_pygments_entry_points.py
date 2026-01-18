from typing import List
import pytest
import pygments.lexers
import pygments.lexer
from IPython.lib.lexers import IPythonConsoleLexer, IPythonLexer, IPython3Lexer
@pytest.mark.parametrize('expected_lexer', EXPECTED_LEXER_NAMES)
def test_pygments_entry_points(expected_lexer: str, all_pygments_lexer_names: List[str]) -> None:
    """Check whether the ``entry_points`` for ``pygments.lexers`` are correct."""
    assert expected_lexer in all_pygments_lexer_names