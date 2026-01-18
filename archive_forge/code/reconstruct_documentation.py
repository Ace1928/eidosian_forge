from typing import Dict, Callable, Iterable, Optional
from .lark import Lark
from .tree import Tree, ParseTree
from .visitors import Transformer_InPlace
from .lexer import Token, PatternStr, TerminalDef
from .grammar import Terminal, NonTerminal, Symbol
from .tree_matcher import TreeMatcher, is_discarded_terminal
from .utils import is_id_continue

    A Reconstructor that will, given a full parse Tree, generate source code.

    Note:
        The reconstructor cannot generate values from regexps. If you need to produce discarded
        regexes, such as newlines, use `term_subs` and provide default values for them.

    Parameters:
        parser: a Lark instance
        term_subs: a dictionary of [Terminal name as str] to [output text as str]
    