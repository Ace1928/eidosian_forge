from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class DirectiveDefinitionNode(TypeSystemDefinitionNode):
    __slots__ = ('description', 'name', 'arguments', 'repeatable', 'locations')
    description: Optional[StringValueNode]
    name: NameNode
    arguments: Tuple[InputValueDefinitionNode, ...]
    repeatable: bool
    locations: Tuple[NameNode, ...]