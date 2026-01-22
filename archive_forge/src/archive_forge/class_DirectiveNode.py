from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class DirectiveNode(Node):
    __slots__ = ('name', 'arguments')
    name: NameNode
    arguments: Tuple[ArgumentNode, ...]