from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class SegmentSum(Operation):

    def __init__(self, num_segments=None, sorted=False):
        super().__init__()
        self.num_segments = num_segments
        self.sorted = sorted

    def compute_output_spec(self, data, segment_ids):
        num_segments = self.num_segments
        output_shape = (num_segments,) + tuple(data.shape[1:])
        return KerasTensor(shape=output_shape, dtype=data.dtype)

    def call(self, data, segment_ids):
        return backend.math.segment_sum(data, segment_ids, num_segments=self.num_segments, sorted=self.sorted)