import numpy as np
from .... import symbol
from .... import ndarray as nd
from ....base import string_types
from ._import_helper import _convert_map as convert_map
def _convert_operator(self, node_name, op_name, attrs, inputs):
    """Convert from onnx operator to mxnet operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        :param node_name : str
            name of the node to be translated.
        :param op_name : str
            Operator name, such as Convolution, FullyConnected
        :param attrs : dict
            Dict of operator attributes
        :param inputs: list
            list of inputs to the operator
        Returns
        -------
        :return mxnet_sym
            Converted mxnet symbol
        """
    if op_name in convert_map:
        op_name, new_attrs, inputs = convert_map[op_name](attrs, inputs, self)
    else:
        raise NotImplementedError('Operator {} not implemented.'.format(op_name))
    if isinstance(op_name, string_types):
        new_op = getattr(symbol, op_name, None)
        if not new_op:
            raise RuntimeError('Unable to map op_name {} to sym'.format(op_name))
        if node_name is None:
            mxnet_sym = new_op(*inputs, **new_attrs)
        else:
            mxnet_sym = new_op(*inputs, name=node_name, **new_attrs)
        return mxnet_sym
    return op_name