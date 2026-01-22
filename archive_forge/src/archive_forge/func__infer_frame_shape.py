import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import util_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis):
    """Infers the shape of the return value of `frame`."""
    frame_length = tensor_util.constant_value(frame_length)
    frame_step = tensor_util.constant_value(frame_step)
    axis = tensor_util.constant_value(axis)
    if signal.shape.ndims is None:
        return None
    if axis is None:
        return [None] * (signal.shape.ndims + 1)
    signal_shape = signal.shape.as_list()
    num_frames = None
    frame_axis = signal_shape[axis]
    outer_dimensions = signal_shape[:axis]
    inner_dimensions = signal_shape[axis:][1:]
    if signal_shape and frame_axis is not None:
        if frame_step is not None and pad_end:
            num_frames = max(0, -(-frame_axis // frame_step))
        elif frame_step is not None and frame_length is not None:
            assert not pad_end
            num_frames = max(0, (frame_axis - frame_length + frame_step) // frame_step)
    return outer_dimensions + [num_frames, frame_length] + inner_dimensions