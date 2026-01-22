from pyparsing import Word, nums, alphas, Combine, oneOf, \
class EvalMultOp(object):
    """Class to evaluate multiplication and division expressions"""

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        prod = self.value[0].eval()
        for op, val in operatorOperands(self.value[1:]):
            if op == '*':
                prod *= val.eval()
            if op == '/':
                prod /= val.eval()
        return prod