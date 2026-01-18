from __future__ import (absolute_import, division, print_function)
import os
import sys
def version_to_str(version):
    """Return a version string from a version tuple."""
    return '.'.join((str(n) for n in version))