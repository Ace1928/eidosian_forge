from warnings import warn
from passlib.context import LazyCryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib import registry
from passlib.utils import has_crypt, unix_crypt_schemes
def _iter_os_crypt_schemes():
    """helper which iterates over supported os_crypt schemes"""
    out = registry.get_supported_os_crypt_schemes()
    if out:
        out += ('unix_disabled',)
    return out