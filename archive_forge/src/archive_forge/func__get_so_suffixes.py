import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _get_so_suffixes():
    suffixes = _extension_suffixes()
    if not suffixes:
        if sys.platform == 'win32':
            suffixes = ['.pyd']
        else:
            suffixes = ['.so']
    return suffixes