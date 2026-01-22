import os
import sys
from itertools import product, starmap
import distutils.command.install_lib as orig
@staticmethod
def _all_packages(pkg_name):
    """
        >>> list(install_lib._all_packages('foo.bar.baz'))
        ['foo.bar.baz', 'foo.bar', 'foo']
        """
    while pkg_name:
        yield pkg_name
        pkg_name, sep, child = pkg_name.rpartition('.')