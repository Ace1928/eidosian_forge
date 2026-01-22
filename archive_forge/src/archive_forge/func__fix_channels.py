from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_channels(op_name, attrs, inputs, proto_obj):
    """A workaround for getting 'channels' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    weight_name = inputs[1].name
    if not weight_name in proto_obj._params:
        raise ValueError('Unable to get channels/units attr from onnx graph.')
    wshape = proto_obj._params[weight_name].shape
    assert len(wshape) >= 2, 'Weights shape is invalid: {}'.format(wshape)
    if op_name == 'FullyConnected':
        attrs['num_hidden'] = wshape[0]
    elif op_name == 'Convolution':
        attrs['num_filter'] = wshape[0]
    elif op_name == 'Deconvolution':
        attrs['num_filter'] = wshape[1]
    return attrs