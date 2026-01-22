import pytest
from mako.ext.beaker_cache import has_beaker
from mako.util import update_wrapper
def _pygments_version():
    try:
        import pygments
        version = pygments.__version__
    except:
        version = '0'
    return version