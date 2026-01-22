from typing import TYPE_CHECKING, Callable, Optional, List, Any
from collections import deque
from ..lexer import Token
from ..tree import Tree
from ..exceptions import UnexpectedEOF, UnexpectedToken
from ..utils import logger, OrderedSet
from .grammar_analysis import GrammarAnalyzer
from ..grammar import NonTerminal
from .earley_common import Item
from .earley_forest import ForestSumVisitor, SymbolNode, StableSymbolNode, TokenNode, ForestToParseTree
The core Earley Scanner.

            This is a custom implementation of the scanner that uses the
            Lark lexer to match tokens. The scan list is built by the
            Earley predictor, based on the previously completed tokens.
            This ensures that at each phase of the parse we have a custom
            lexer context, allowing for more complex ambiguities.