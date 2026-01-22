from typing import Iterable, List, Tuple
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode, Token, Tree
from _qpd_antlr import QPDLexer, QPDParser
def _to_cased_code(code: str, rule: str, ansi_sql: bool) -> Tuple[str, Tree]:
    tree = _to_tree(code.upper(), rule, True, ansi_sql=ansi_sql)
    tokens = [t for t in _to_tokens(tree) if _is_keyword(t)]
    start = 0
    cased_code: List[str] = []
    for t in tokens:
        if t.start > start:
            cased_code.append(code[start:t.start])
        cased_code.append(code[t.start:t.stop + 1].upper())
        start = t.stop + 1
    if start < len(code):
        cased_code.append(code[start:])
    return (''.join(cased_code), tree)