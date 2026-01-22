import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
class BindLocation:
    """A class for storing the bind location for a Cheroot instance."""