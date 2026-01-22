from typing import Iterable, List, Tuple
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode, Token, Tree
from _qpd_antlr import QPDLexer, QPDParser
class _ErrorListener(ErrorListener):

    def __init__(self, lines: List[str]):
        super().__init__()
        self._lines = lines

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f'{msg}\nline {line}: {self._lines[line - 1]}\n{offendingSymbol}')

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        pass

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        pass

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        pass