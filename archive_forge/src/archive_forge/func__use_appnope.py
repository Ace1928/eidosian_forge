import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def _use_appnope():
    """Should we use appnope for dealing with OS X app nap?

    Checks if we are on OS X 10.9 or greater.
    """
    return sys.platform == 'darwin' and V(platform.mac_ver()[0]) >= V('10.9')