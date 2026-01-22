import inspect
import os
import warnings
import param
from packaging.version import Version
class HoloviewsUserWarning(UserWarning):
    """A Holoviews-specific ``UserWarning`` subclass.
    Used to selectively filter Holoviews warnings for unconditional display.
    """