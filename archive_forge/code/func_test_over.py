from __future__ import annotations
import numpy as np
from datashader.composite import add, saturate, over, source
def test_over():
    o = src.copy()
    o[0, 1] = 0
    np.testing.assert_equal(over(src, clear), o)
    np.testing.assert_equal(over(src, clear_white), o)
    o = np.array([[4294967295, 4294967295, 4294967295], [4294935170, 4286775170, 4286743295], [4294901760, 4278190080, 4292006610]])
    np.testing.assert_equal(over(src, white), o)
    o = np.array([[4294901760, 4294901760, 4294967295], [4294901760, 4286741760, 4286709885], [4294901760, 4278190080, 4291955981]])
    np.testing.assert_equal(over(src, blue), o)
    o = np.array([[2113863680, 2113863680, 4294967295], [3170828288, 3159795712, 3159752872], [4294901760, 4278190080, 2595558934]])
    np.testing.assert_equal(over(src, half_blue), o)
    o = np.array([[2105344125, 2105344125, 4294967295], [3167944746, 3156912170, 3156869331], [4294901760, 4278190080, 2590250596]])
    np.testing.assert_equal(over(src, half_purple), o)