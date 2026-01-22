from copy import copy, deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
from .source import Source
from .token_kind import TokenKind
from ..pyutils import camel_to_snake
class EnumValueDefinitionNode(DefinitionNode):
    __slots__ = ('description', 'name', 'directives')
    description: Optional[StringValueNode]
    name: NameNode
    directives: Tuple[ConstDirectiveNode, ...]