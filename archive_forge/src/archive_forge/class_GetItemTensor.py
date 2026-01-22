from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class GetItemTensor(Constraint):

    def __init__(self, tensor_size, index_tuple, res, input_var):
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        assert isinstance(res, TVar)
        self.res = res
        self.tensor_size = tensor_size
        self.index_tuple = index_tuple
        self.input_var = input_var

    def __repr__(self):
        return f' {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})'

    def __eq__(self, other):
        if isinstance(other, GetItemTensor):
            return self.res == other.res and self.tensor_size == other.tensor_size and (self.index_tuple == other.index_tuple) and (self.input_var == other.input_var)
        else:
            return False