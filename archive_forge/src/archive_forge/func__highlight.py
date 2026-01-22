import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
def _highlight(self, source: str, lexer: Literal['diff', 'python']='python') -> str:
    """Highlight the given source if we have markup support."""
    from _pytest.config.exceptions import UsageError
    if not source or not self.hasmarkup or (not self.code_highlight):
        return source
    try:
        from pygments.formatters.terminal import TerminalFormatter
        if lexer == 'python':
            from pygments.lexers.python import PythonLexer as Lexer
        elif lexer == 'diff':
            from pygments.lexers.diff import DiffLexer as Lexer
        from pygments import highlight
        import pygments.util
    except ImportError:
        return source
    else:
        try:
            highlighted: str = highlight(source, Lexer(), TerminalFormatter(bg=os.getenv('PYTEST_THEME_MODE', 'dark'), style=os.getenv('PYTEST_THEME')))
            if highlighted[-1] == '\n' and source[-1] != '\n':
                highlighted = highlighted[:-1]
            return '\x1b[0m' + highlighted
        except pygments.util.ClassNotFound as e:
            raise UsageError("PYTEST_THEME environment variable had an invalid value: '{}'. Only valid pygment styles are allowed.".format(os.getenv('PYTEST_THEME'))) from e
        except pygments.util.OptionError as e:
            raise UsageError("PYTEST_THEME_MODE environment variable had an invalid value: '{}'. The only allowed values are 'dark' and 'light'.".format(os.getenv('PYTEST_THEME_MODE'))) from e