import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def _annotate_ndarray_lines(array_lines, tensor, np_printoptions=None, offset=0):
    """Generate annotations for line-by-line begin indices of tensor text.

  Parse the numpy-generated text representation of a numpy ndarray to
  determine the indices of the first element of each text line (if any
  element is present in the line).

  For example, given the following multi-line ndarray text representation:
      ["array([[ 0.    ,  0.0625,  0.125 ,  0.1875],",
       "       [ 0.25  ,  0.3125,  0.375 ,  0.4375],",
       "       [ 0.5   ,  0.5625,  0.625 ,  0.6875],",
       "       [ 0.75  ,  0.8125,  0.875 ,  0.9375]])"]
  the generate annotation will be:
      {0: {BEGIN_INDICES_KEY: [0, 0]},
       1: {BEGIN_INDICES_KEY: [1, 0]},
       2: {BEGIN_INDICES_KEY: [2, 0]},
       3: {BEGIN_INDICES_KEY: [3, 0]}}

  Args:
    array_lines: Text lines representing the tensor, as a list of str.
    tensor: The tensor being formatted as string.
    np_printoptions: A dictionary of keyword arguments that are passed to a
      call of np.set_printoptions().
    offset: Line number offset applied to the line indices in the returned
      annotation.

  Returns:
    An annotation as a dict.
  """
    if np_printoptions and 'edgeitems' in np_printoptions:
        edge_items = np_printoptions['edgeitems']
    else:
        edge_items = _NUMPY_DEFAULT_EDGE_ITEMS
    annotations = {}
    annotations['tensor_metadata'] = {'dtype': tensor.dtype, 'shape': tensor.shape}
    dims = np.shape(tensor)
    ndims = len(dims)
    if ndims == 0:
        return annotations
    curr_indices = [0] * len(dims)
    curr_dim = 0
    for i, raw_line in enumerate(array_lines):
        line = raw_line.strip()
        if not line:
            continue
        if line == _NUMPY_OMISSION:
            annotations[offset + i] = {OMITTED_INDICES_KEY: copy.copy(curr_indices)}
            curr_indices[curr_dim - 1] = dims[curr_dim - 1] - edge_items
        else:
            num_lbrackets = line.count('[')
            num_rbrackets = line.count(']')
            curr_dim += num_lbrackets - num_rbrackets
            annotations[offset + i] = {BEGIN_INDICES_KEY: copy.copy(curr_indices)}
            if num_rbrackets == 0:
                line_content = line[line.rfind('[') + 1:]
                num_elements = line_content.count(',')
                curr_indices[curr_dim - 1] += num_elements
            elif curr_dim > 0:
                curr_indices[curr_dim - 1] += 1
                for k in range(curr_dim, ndims):
                    curr_indices[k] = 0
    return annotations