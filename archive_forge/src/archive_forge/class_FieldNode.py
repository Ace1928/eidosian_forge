from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class FieldNode(SelectionNode):
    __slots__ = ('alias', 'name', 'arguments', 'selection_set')
    alias: Optional[NameNode]
    name: NameNode
    arguments: Tuple['ArgumentNode', ...]
    selection_set: Optional[SelectionSetNode]