import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
def test_optical_flow_dtype():
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    flow_f64 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float64)
    assert flow_f64.dtype == np.float64
    flow_f32 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float32)
    assert flow_f32.dtype == np.float32
    assert np.abs(flow_f64 - flow_f32).mean() < 0.001