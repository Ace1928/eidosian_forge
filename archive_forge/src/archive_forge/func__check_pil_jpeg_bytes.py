from binascii import a2b_base64
from io import BytesIO
import pytest
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import _PNG, _JPEG
from .. import pylabtools as pt
from IPython.testing import decorators as dec
def _check_pil_jpeg_bytes():
    """Skip if PIL can't write JPEGs to BytesIO objects"""
    from PIL import Image
    buf = BytesIO()
    img = Image.new('RGB', (4, 4))
    try:
        img.save(buf, 'jpeg')
    except Exception as e:
        ename = e.__class__.__name__
        raise pytest.skip("PIL can't write JPEG to BytesIO: %s: %s" % (ename, e)) from e