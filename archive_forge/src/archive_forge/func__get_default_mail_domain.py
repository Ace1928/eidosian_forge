import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _get_default_mail_domain(mailname_file='/etc/mailname'):
    """If possible, return the assumed default email domain.

    :returns: string mail domain, or None.
    """
    if sys.platform == 'win32':
        return None
    try:
        f = open(mailname_file)
    except OSError:
        return None
    try:
        domain = f.readline().strip()
        return domain
    finally:
        f.close()