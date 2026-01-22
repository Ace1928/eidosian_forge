from __future__ import annotations
import inspect
import os
import warnings
import param
from packaging.version import Version
class PanelUserWarning(UserWarning):
    """A Panel-specific ``UserWarning`` subclass.
    Used to selectively filter Panel warnings for unconditional display.
    """