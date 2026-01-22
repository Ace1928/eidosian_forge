import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _modname_to_file(outputdir, modname, extension):
    parts = modname.split('.')
    try:
        os.makedirs(os.path.join(outputdir, *parts[:-1]))
    except OSError:
        pass
    parts[-1] += extension
    return (os.path.join(outputdir, *parts), parts)