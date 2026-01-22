from io import BytesIO
import functools
from fontTools import subset
import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase
@functools.lru_cache(50)
def _cached_get_afm_from_fname(fname):
    with open(fname, 'rb') as fh:
        return AFM(fh)