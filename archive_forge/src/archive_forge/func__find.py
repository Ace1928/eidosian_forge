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
def _find(name: str, ns: dict):
    """Find a Python name, which might include dot-separated path to a name.

    Args:
    - name: a key in dict if no dot ('.') in it, otherwise
    a sequence of dot-separated namespaces with the name of the
    object last (e.g., `package.module.name`).
    Returns:
    The object wanted. Raises a NameError or an AttributeError if not found.
    """
    obj = None
    obj_path = name.split('.')
    look_for_i = 0
    try:
        obj = ns[obj_path[look_for_i]]
    except KeyError as e:
        message = f"name '{obj_path[look_for_i]}' is not defined."
        if obj_path[look_for_i] == '':
            message += ' Did you forget to remove trailing comma `,` or included spaces?'
        raise NameError(message) from e
    look_for_i += 1
    while look_for_i < len(obj_path):
        try:
            obj = getattr(obj, obj_path[look_for_i])
        except AttributeError as e:
            raise AttributeError(f"'{'.'.join(obj_path[:look_for_i])}' has no attribute '{obj_path[look_for_i]}'.") from e
        look_for_i += 1
    return obj