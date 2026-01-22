import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class Base(NodeStateTracker, gast.NodeTransformer):
    """Base class for general-purpose Python-to-Python code transformation.

  This is an extension of ast.NodeTransformer that provides the additional
  functions offered by NodeStateTracker.
  """

    def create_assignment(self, target, expression):
        template = '\n      target = expression\n    '
        return templates.replace(template, target=target, expression=expression)

    def apply_to_single_assignments(self, targets, values, apply_fn):
        """Applies a function to each individual assignment.

    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.

    Examples:

    The following will result in apply_fn(a, c), apply_fn(b, d):

        a, b = c, d

    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

        a, b = c

    The following will result in apply_fn(a, (b, c)):

        a = b, c

    It uses the visitor pattern to allow subclasses to process single
    assignments individually.

    Args:
      targets: list, tuple of or individual AST node. Should be used with the
        targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
        respective nodes of each single assignment. The signature is
        apply_fn(target, value), no return value.
    """
        if not isinstance(targets, (list, tuple)):
            targets = (targets,)
        for target in targets:
            if isinstance(target, (gast.Tuple, gast.List)):
                for i in range(len(target.elts)):
                    target_el = target.elts[i]
                    if isinstance(values, (gast.Tuple, gast.List)):
                        value_el = values.elts[i]
                    else:
                        value_el = gast.Subscript(values, i, ctx=gast.Store())
                    self.apply_to_single_assignments(target_el, value_el, apply_fn)
            else:
                apply_fn(target, values)

    def visit(self, node):
        if not isinstance(node, gast.AST):
            msg = 'invalid value for "node": expected "ast.AST", got "{}"; to visit lists of nodes, use "visit_block" instead'.format(type(node))
            raise ValueError(msg)
        if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
            return node
        parent_origin = self.ctx.current_origin
        if anno.hasanno(node, anno.Basic.ORIGIN):
            self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)
        try:
            processing_expr_node = isinstance(node, gast.Expr)
            if processing_expr_node:
                entry_expr_value = node.value
            result = super(Base, self).visit(node)
            if processing_expr_node and isinstance(result, gast.Expr) and (result.value is not entry_expr_value):
                if isinstance(result.value, (list, tuple, gast.Assign, gast.AugAssign)):
                    result = result.value
            if result is not node and result is not None:
                inherited_origin = anno.getanno(node, anno.Basic.ORIGIN, default=parent_origin)
                if inherited_origin is not None:
                    nodes_to_adjust = result
                    if isinstance(result, (list, tuple)):
                        nodes_to_adjust = result
                    else:
                        nodes_to_adjust = (result,)
                    for n in nodes_to_adjust:
                        if not anno.hasanno(n, anno.Basic.ORIGIN):
                            anno.setanno(n, anno.Basic.ORIGIN, inherited_origin)
        finally:
            self.ctx.current_origin = parent_origin
        return result