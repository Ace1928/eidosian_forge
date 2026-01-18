import doctest
import os
import sys
from io import StringIO
import breezy
from .. import bedding, crash, osutils, plugin, tests
from . import features
Reporting of crash-type bugs without apport.

    This should work in all environments.
    