import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import templates
class AssertTransformer(converter.Base):
    """Transforms Assert nodes to Call so they can be handled as functions."""

    def visit_Assert(self, node):
        self.generic_visit(node)
        template = '\n      ag__.assert_stmt(test, lambda: msg)\n    '
        if node.msg is None:
            return templates.replace(template, test=node.test, msg=gast.Constant('Assertion error', kind=None))
        elif isinstance(node.msg, gast.Constant):
            return templates.replace(template, test=node.test, msg=node.msg)
        else:
            raise NotImplementedError('can only convert string messages for now.')