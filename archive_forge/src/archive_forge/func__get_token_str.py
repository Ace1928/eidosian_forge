from typing import Iterable, List, Optional, Tuple
import pkg_resources
from antlr4 import InputStream, Token
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode
from packaging import version
from fugue_sql_antlr._parser.fugue_sqlParser import fugue_sqlParser
from fugue_sql_antlr._parser.sa_fugue_sql import (
from fugue_sql_antlr.constants import (
def _get_token_str(token: Token):
    s = self.code[token.start:token.stop + 1]
    if upper_keyword and self._ignore_case and hasattr(token, 'is_keyword'):
        return s.upper()
    return s