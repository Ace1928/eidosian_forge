from typing import Iterable, List, Optional, Tuple
import pkg_resources
from antlr4 import InputStream, Token
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode
from packaging import version
from fugue_sql_antlr._parser.fugue_sqlParser import fugue_sqlParser
from fugue_sql_antlr._parser.sa_fugue_sql import (
from fugue_sql_antlr.constants import (
def _detect_case_issue(text: str, lower_case_percentage: float) -> bool:
    letters, lower = (0, 0.0)
    for c in text:
        if 'a' <= c <= 'z':
            lower += 1.0
            letters += 1
        elif 'A' <= c <= 'Z':
            letters += 1
    if letters == 0:
        return False
    return lower / letters >= lower_case_percentage