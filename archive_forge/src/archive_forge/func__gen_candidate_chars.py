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
def _gen_candidate_chars(characters):
    """Generate a string containing all valid chars as defined by ``characters``

    :arg characters: A list of character specs. The character specs are
        shorthand names for sets of characters like 'digits', 'ascii_letters',
        or 'punctuation' or a string to be included verbatim.

    The values of each char spec can be:

    * a name of an attribute in the 'strings' module ('digits' for example).
      The value of the attribute will be added to the candidate chars.
    * a string of characters. If the string isn't an attribute in 'string'
      module, the string will be directly added to the candidate chars.

    For example::

        characters=['digits', '?|']``

    will match ``string.digits`` and add all ascii digits.  ``'?|'`` will add
    the question mark and pipe characters directly. Return will be the string::

        u'0123456789?|'
    """
    chars = []
    for chars_spec in characters:
        chars.append(to_text(getattr(string, to_native(chars_spec), chars_spec), errors='strict'))
    chars = u''.join(chars).replace(u'"', u'').replace(u"'", u'')
    return chars