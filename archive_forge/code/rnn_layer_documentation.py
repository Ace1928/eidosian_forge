import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
 forward using CUDNN or CPU kenrel