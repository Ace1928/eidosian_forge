from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class TVar:
    """
    Tensor variable with no tensor constructor
    """

    def __init__(self, tvar):
        """
        :param tvar: tensor variable
        """
        self.tvar = tvar

    def __repr__(self):
        return f'TV({self.tvar})'

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False