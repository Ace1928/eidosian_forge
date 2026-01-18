import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_roundtrip_conversion():
    import tensorflow as tf
    ops = get_current_ops()
    xp_tensor = ops.alloc2f(2, 3, zeros=True)
    tf_tensor = xp2tensorflow(xp_tensor)
    assert isinstance(tf_tensor, tf.Tensor)
    new_xp_tensor = tensorflow2xp(tf_tensor, ops=ops)
    assert ops.xp.array_equal(xp_tensor, new_xp_tensor)