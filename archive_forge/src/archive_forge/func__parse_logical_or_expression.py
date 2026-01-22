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
def _parse_logical_or_expression(self) -> ASTExpression:

    def _parse_bin_op_expr(self, opId):
        if opId + 1 == len(_expression_bin_ops):

            def parser() -> ASTExpression:
                return self._parse_cast_expression()
        else:

            def parser() -> ASTExpression:
                return _parse_bin_op_expr(self, opId + 1)
        exprs = []
        ops = []
        exprs.append(parser())
        while True:
            self.skip_ws()
            pos = self.pos
            oneMore = False
            for op in _expression_bin_ops[opId]:
                if op[0] in 'abcnox':
                    if not self.skip_word(op):
                        continue
                elif not self.skip_string(op):
                    continue
                if op == '&' and self.current_char == '&':
                    self.pos -= 1
                    break
                try:
                    expr = parser()
                    exprs.append(expr)
                    ops.append(op)
                    oneMore = True
                    break
                except DefinitionError:
                    self.pos = pos
            if not oneMore:
                break
        return ASTBinOpExpr(exprs, ops)
    return _parse_bin_op_expr(self, 0)