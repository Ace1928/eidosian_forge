import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_str_transform():
    assert str(plt.subplot(projection='polar').transData) == 'CompositeGenericTransform(\n    CompositeGenericTransform(\n        CompositeGenericTransform(\n            TransformWrapper(\n                BlendedAffine2D(\n                    IdentityTransform(),\n                    IdentityTransform())),\n            CompositeAffine2D(\n                Affine2D().scale(1.0),\n                Affine2D().scale(1.0))),\n        PolarTransform(\n            PolarAxes(0.125,0.1;0.775x0.8),\n            use_rmin=True,\n            _apply_theta_transforms=False)),\n    CompositeGenericTransform(\n        CompositeGenericTransform(\n            PolarAffine(\n                TransformWrapper(\n                    BlendedAffine2D(\n                        IdentityTransform(),\n                        IdentityTransform())),\n                LockableBbox(\n                    Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),\n                    [[-- --]\n                     [-- --]])),\n            BboxTransformFrom(\n                _WedgeBbox(\n                    (0.5, 0.5),\n                    TransformedBbox(\n                        Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),\n                        CompositeAffine2D(\n                            Affine2D().scale(1.0),\n                            Affine2D().scale(1.0))),\n                    LockableBbox(\n                        Bbox(x0=0.0, y0=0.0, x1=6.283185307179586, y1=1.0),\n                        [[-- --]\n                         [-- --]])))),\n        BboxTransformTo(\n            TransformedBbox(\n                Bbox(x0=0.125, y0=0.09999999999999998, x1=0.9, y1=0.9),\n                BboxTransformTo(\n                    TransformedBbox(\n                        Bbox(x0=0.0, y0=0.0, x1=8.0, y1=6.0),\n                        Affine2D().scale(80.0)))))))'