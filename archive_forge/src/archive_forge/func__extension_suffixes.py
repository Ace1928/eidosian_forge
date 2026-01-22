import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _extension_suffixes():
    return [suffix for suffix, _, type in imp.get_suffixes() if type == imp.C_EXTENSION]