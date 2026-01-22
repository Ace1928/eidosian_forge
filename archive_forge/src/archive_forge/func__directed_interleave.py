from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def _directed_interleave(selector_input, data_inputs, stop_on_empty_dataset=False):
    return _DirectedInterleaveDataset(selector_input, data_inputs, stop_on_empty_dataset=stop_on_empty_dataset)