import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def _list_cmaps(provider=None, records=False):
    """
    List available colormaps by combining matplotlib, bokeh, and
    colorcet colormaps or palettes if available. May also be
    narrowed down to a particular provider or list of providers.
    """
    if provider is None:
        provider = providers
    elif isinstance(provider, str):
        if provider not in providers:
            raise ValueError(f'Colormap provider {provider!r} not recognized, must be one of {providers!r}')
        provider = [provider]
    cmaps = []

    def info(provider, names):
        return [CMapInfo(name=n, provider=provider, category=None, source=None, bg=None) for n in names] if records else list(names)
    if 'matplotlib' in provider:
        try:
            import matplotlib as mpl
            from matplotlib import cm
            if hasattr(mpl, 'colormaps'):
                mpl_cmaps = list(mpl.colormaps)
            elif hasattr(cm, '_cmap_registry'):
                mpl_cmaps = list(cm._cmap_registry)
            else:
                mpl_cmaps = list(cm.cmaps_listed) + list(cm.datad)
            cmaps += info('matplotlib', mpl_cmaps)
            cmaps += info('matplotlib', [cmap + '_r' for cmap in mpl_cmaps if not cmap.endswith('_r')])
        except ImportError:
            pass
    if 'bokeh' in provider:
        try:
            from bokeh import palettes
            cmaps += info('bokeh', palettes.all_palettes)
            cmaps += info('bokeh', [p + '_r' for p in palettes.all_palettes if not p.endswith('_r')])
        except ImportError:
            pass
    if 'colorcet' in provider:
        try:
            from colorcet import glasbey_hv, palette_n
            cet_maps = palette_n.copy()
            cet_maps['glasbey_hv'] = glasbey_hv
            cmaps += info('colorcet', cet_maps)
            cmaps += info('colorcet', [p + '_r' for p in cet_maps if not p.endswith('_r')])
        except ImportError:
            pass
    return sorted(unique_iterator(cmaps))