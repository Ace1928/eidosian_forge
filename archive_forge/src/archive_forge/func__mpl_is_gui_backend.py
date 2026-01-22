from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def _mpl_is_gui_backend():
    try:
        backend = mpl.get_backend()
    except Exception:
        return False
    if backend in ['module://ipykernel.pylab.backend_inline', 'agg']:
        return False
    else:
        return True