import os
import sys
from itertools import product, starmap
import distutils.command.install_lib as orig
@staticmethod
def _gen_exclusion_paths():
    """
        Generate file paths to be excluded for namespace packages (bytecode
        cache files).
        """
    yield '__init__.py'
    yield '__init__.pyc'
    yield '__init__.pyo'
    if not hasattr(sys, 'implementation'):
        return
    base = os.path.join('__pycache__', '__init__.' + sys.implementation.cache_tag)
    yield (base + '.pyc')
    yield (base + '.pyo')
    yield (base + '.opt-1.pyc')
    yield (base + '.opt-2.pyc')