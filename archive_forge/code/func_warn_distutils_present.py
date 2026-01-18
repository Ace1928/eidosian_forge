import sys
import os
def warn_distutils_present():
    if 'distutils' not in sys.modules:
        return
    import warnings
    warnings.warn('Distutils was imported before Setuptools, but importing Setuptools also replaces the `distutils` module in `sys.modules`. This may lead to undesirable behaviors or errors. To avoid these issues, avoid using distutils directly, ensure that setuptools is installed in the traditional way (e.g. not an editable install), and/or make sure that setuptools is always imported before distutils.')