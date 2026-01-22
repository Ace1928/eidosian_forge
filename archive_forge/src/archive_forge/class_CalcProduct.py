from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """

    def __init__(self, start, end, flattened, dims_to_flatten):
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        assert isinstance(dims_to_flatten, list)
        assert isinstance(flattened, TVar)
        assert isinstance(start, int)
        assert isinstance(end, int)
        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    def __eq__(self, other):
        if isinstance(other, CalcProduct):
            return self.start == other.start and self.end == other.end and (self.dims_to_flatten == other.dims_to_flatten) and (self.flattened == other.flattened)
        else:
            return False

    def __repr__(self):
        return f'{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})'