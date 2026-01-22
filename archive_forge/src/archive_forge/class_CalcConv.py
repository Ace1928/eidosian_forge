from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class CalcConv(Constraint):

    def __init__(self, conv_result, input_var, c_out, kernel, padding, stride, dilation, matching_constraint_vars):
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output chanel type
        :param kernel: kernel tuple
        """
        self.conv_result = conv_result
        self.input_var = input_var
        self.c_out = c_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self):
        return f'{self.conv_result} = calc-conv({self.input_var}, {self.c_out}, {self.kernel}, {self.padding}, {self.stride}, {self.dilation})'

    def __eq__(self, other):
        if isinstance(other, CalcConv):
            return self.conv_result == other.conv_result and self.input_var == other.input_var and (self.c_out == other.c_out) and (self.kernel == other.kernel) and (self.padding == other.padding) and (self.stride == other.stride) and (self.dilation == other.dilation) and (self.matching_constraint == other.matching_constraint)
        else:
            return False