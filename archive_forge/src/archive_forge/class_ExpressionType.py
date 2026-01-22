from inspect import isclass
class ExpressionType(Type):
    """
            Result type of an operator call.
            """

    def __init__(self, op, *exprs):
        super(ExpressionType, self).__init__(op=op, exprs=exprs)

    def iscombined(self):
        return any((expr.iscombined() for expr in self.exprs))

    def generate(self, ctx):
        gexprs = ['std::declval<{0}>()'.format(ctx(expr)) for expr in self.exprs]
        return 'decltype({0})'.format(self.op(*gexprs))