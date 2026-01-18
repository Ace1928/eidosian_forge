import io
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, TiffTags
import pytest
from matplotlib import (
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
def test_pil_kwargs_tiff():
    buf = io.BytesIO()
    pil_kwargs = {'description': 'test image'}
    plt.figure().savefig(buf, format='tiff', pil_kwargs=pil_kwargs)
    im = Image.open(buf)
    tags = {TiffTags.TAGS_V2[k].name: v for k, v in im.tag_v2.items()}
    assert tags['ImageDescription'] == 'test image'