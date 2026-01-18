from __future__ import (absolute_import, division, print_function)
import os
import shutil
from errno import EEXIST
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def unfrackpath(path, follow=True, basedir=None):
    """
    Returns a path that is free of symlinks (if follow=True), environment variables, relative path traversals and symbols (~)

    :arg path: A byte or text string representing a path to be canonicalized
    :arg follow: A boolean to indicate of symlinks should be resolved or not
    :raises UnicodeDecodeError: If the canonicalized version of the path
        contains non-utf8 byte sequences.
    :rtype: A text string (unicode on pyyhon2, str on python3).
    :returns: An absolute path with symlinks, environment variables, and tilde
        expanded.  Note that this does not check whether a path exists.

    example::
        '$HOME/../../var/mail' becomes '/var/spool/mail'
    """
    b_basedir = to_bytes(basedir, errors='surrogate_or_strict', nonstring='passthru')
    if b_basedir is None:
        b_basedir = to_bytes(os.getcwd(), errors='surrogate_or_strict')
    elif os.path.isfile(b_basedir):
        b_basedir = os.path.dirname(b_basedir)
    b_final_path = os.path.expanduser(os.path.expandvars(to_bytes(path, errors='surrogate_or_strict')))
    if not os.path.isabs(b_final_path):
        b_final_path = os.path.join(b_basedir, b_final_path)
    if follow:
        b_final_path = os.path.realpath(b_final_path)
    return to_text(os.path.normpath(b_final_path), errors='surrogate_or_strict')