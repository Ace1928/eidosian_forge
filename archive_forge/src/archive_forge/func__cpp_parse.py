import sys
import types
import antlr4
from antlr4 import InputStream, CommonTokenStream, Token
from antlr4.tree.Tree import ParseTree
from antlr4.error.ErrorListener import ErrorListener
from .fugue_sqlParser import fugue_sqlParser
from .fugue_sqlLexer import fugue_sqlLexer
def _cpp_parse(stream: InputStream, entry_rule_name: str, sa_err_listener: SA_ErrorListener=None) -> ParseTree:
    if not isinstance(stream, InputStream):
        raise TypeError("'stream' shall be an Antlr InputStream")
    if not isinstance(entry_rule_name, str):
        raise TypeError("'entry_rule_name' shall be a string")
    if sa_err_listener is not None and (not isinstance(sa_err_listener, SA_ErrorListener)):
        raise TypeError("'sa_err_listener' shall be an instance of SA_ErrorListener or None")
    return sa_fugue_sql_cpp_parser.do_parse(fugue_sqlParser, stream, entry_rule_name, sa_err_listener)