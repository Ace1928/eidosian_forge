from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import nest as tf_nest
def _concatenate(input_dataset, dataset_to_concatenate, name):
    return _ConcatenateDataset(input_dataset, dataset_to_concatenate, name)