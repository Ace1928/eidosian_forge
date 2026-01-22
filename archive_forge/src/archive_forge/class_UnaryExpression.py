from ..utils import SchemaBase
class UnaryExpression(Expression):

    def __init__(self, op, val):
        super(UnaryExpression, self).__init__(op=op, val=val)

    def __repr__(self):
        return '({op}{val})'.format(op=self.op, val=_js_repr(self.val))