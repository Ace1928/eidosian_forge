import functools
import warnings
class jupyterlab_deprecation(Warning):
    """Create our own deprecation class, since Python >= 2.7
    silences deprecations by default.
    """
    pass