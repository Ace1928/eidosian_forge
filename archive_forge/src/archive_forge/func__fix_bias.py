from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_bias(op_name, attrs, num_inputs):
    """A workaround for 'use_bias' attribute since onnx don't provide this attribute,
    we have to check the number of inputs to decide it."""
    if num_inputs == 3:
        attrs['no_bias'] = False
    elif num_inputs == 2:
        attrs['no_bias'] = True
    else:
        raise ValueError('Unexpected number of inputs for: {}'.format(op_name))
    return attrs