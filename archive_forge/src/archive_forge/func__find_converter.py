import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
def _find_converter(self, name: str, local_ns: dict) -> ro.conversion.Converter:
    converter = None
    if name is None:
        converter = self.converter
    else:
        try:
            converter = _find(name, local_ns)
        except NameError:
            if self.shell is None:
                warnings.warn(f'The shell is None. Unable to look for converter {name}.')
            else:
                converter = _find(name, self.shell.user_ns)
    if not isinstance(converter, Converter):
        raise TypeError("'%s' must be a %s object (but it is a %s)." % (converter, Converter, type(converter)))
    return converter