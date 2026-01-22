import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
class ASTGnuAttribute(ASTBaseBase):

    def __init__(self, name: str, args: Optional['ASTBaseParenExprList']) -> None:
        self.name = name
        self.args = args

    def _stringify(self, transform: StringifyTransform) -> str:
        res = [self.name]
        if self.args:
            res.append(transform(self.args))
        return ''.join(res)