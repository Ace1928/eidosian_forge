from collections import namedtuple
import textwrap
def _callback_use_pygeos(key, value):
    assert key == 'use_pygeos'
    import geopandas._compat as compat
    compat.set_use_pygeos(value)