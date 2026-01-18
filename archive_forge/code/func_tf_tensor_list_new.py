import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def tf_tensor_list_new(elements, element_dtype=None, element_shape=None):
    """Overload of new_list that stages a Tensor list creation."""
    if tensor_util.is_tf_type(elements):
        if element_shape is not None:
            raise ValueError('element shape may not be specified when creating list from tensor')
        element_shape = array_ops.shape(elements)[1:]
        l = list_ops.tensor_list_from_tensor(elements, element_shape=element_shape)
        return l
    elements = tuple((ops.convert_to_tensor(el) for el in elements))
    all_dtypes = set((el.dtype for el in elements))
    if len(all_dtypes) == 1:
        inferred_dtype = tuple(all_dtypes)[0]
        if element_dtype is not None and element_dtype != inferred_dtype:
            raise ValueError('incompatible dtype; specified: {}, inferred from {}: {}'.format(element_dtype, elements, inferred_dtype))
    elif all_dtypes:
        if element_dtype is not None:
            raise ValueError('specified dtype {} is inconsistent with that of elements {}'.format(element_dtype, elements))
        inferred_dtype = dtypes.variant
    else:
        inferred_dtype = dtypes.variant
    all_shapes = set((tuple(el.shape.as_list()) for el in elements))
    if len(all_shapes) == 1:
        inferred_shape = array_ops.shape(elements[0])
        if element_shape is not None and element_shape != inferred_shape:
            raise ValueError('incompatible shape; specified: {}, inferred from {}: {}'.format(element_shape, elements, inferred_shape))
    elif all_shapes:
        if element_shape is not None:
            raise ValueError('specified shape {} is inconsistent with that of elements {}'.format(element_shape, elements))
        inferred_shape = constant_op.constant(-1)
    else:
        inferred_shape = constant_op.constant(-1)
    if element_dtype is None:
        element_dtype = inferred_dtype
    if element_shape is None:
        element_shape = inferred_shape
    element_shape = ops.convert_to_tensor(element_shape, dtype=dtypes.int32)
    l = list_ops.empty_tensor_list(element_shape=element_shape, element_dtype=element_dtype)
    for el in elements:
        l = list_ops.tensor_list_push_back(l, el)
    return l