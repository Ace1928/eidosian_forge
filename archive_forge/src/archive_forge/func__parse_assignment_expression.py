import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
def _parse_assignment_expression(self) -> ASTExpression:
    exprs = []
    ops = []
    orExpr = self._parse_logical_or_expression()
    exprs.append(orExpr)
    while True:
        oneMore = False
        self.skip_ws()
        for op in _expression_assignment_ops:
            if op[0] in 'abcnox':
                if not self.skip_word(op):
                    continue
            elif not self.skip_string(op):
                continue
            expr = self._parse_logical_or_expression()
            exprs.append(expr)
            ops.append(op)
            oneMore = True
        if not oneMore:
            break
    return ASTAssignmentExpr(exprs, ops)