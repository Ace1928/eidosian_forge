import os
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
Sets parameters for SaveDatasetOp and SaveDatasetV2Op.