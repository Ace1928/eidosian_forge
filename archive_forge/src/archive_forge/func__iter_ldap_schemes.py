import logging; log = logging.getLogger(__name__)
from itertools import chain
from passlib import hash
from passlib.context import LazyCryptContext
from passlib.utils import sys_bits
def _iter_ldap_schemes():
    """helper which iterates over supported std ldap schemes"""
    return chain(std_ldap_schemes, _iter_ldap_crypt_schemes())