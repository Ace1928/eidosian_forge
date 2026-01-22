from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class Disj(Constraint):

    def __init__(self, disjuncts):
        """
        :param disjuncts: Disjunction of constraints
        """
        self.disjuncts = disjuncts

    def __eq__(self, other):
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts and self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self):
        return f'Or({self.disjuncts})'