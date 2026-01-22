from __future__ import (absolute_import, division, print_function)
import os
import string
import time
import hashlib
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import string_types
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
from ansible.utils.encrypt import BaseHash, do_encrypt, random_password, random_salt
from ansible.utils.path import makedirs_safe
def _format_content(password, salt, encrypt=None, ident=None):
    """Format the password and salt for saving
    :arg password: the plaintext password to save
    :arg salt: the salt to use when encrypting a password
    :arg encrypt: Which method the user requests that this password is encrypted.
        Note that the password is saved in clear.  Encrypt just tells us if we
        must save the salt value for idempotence.  Defaults to None.
    :arg ident: Which version of BCrypt algorithm to be used.
        Valid only if value of encrypt is bcrypt.
        Defaults to None.
    :returns: a text string containing the formatted information

    .. warning:: Passwords are saved in clear.  This is because the playbooks
        expect to get cleartext passwords from this lookup.
    """
    if not encrypt and (not salt):
        return password
    if not salt:
        raise AnsibleAssertionError('_format_content was called with encryption requested but no salt value')
    if ident:
        return u'%s salt=%s ident=%s' % (password, salt, ident)
    return u'%s salt=%s' % (password, salt)