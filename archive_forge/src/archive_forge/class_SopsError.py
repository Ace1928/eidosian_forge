from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
class SopsError(Exception):
    """ Extend Exception class with sops specific information """

    def __init__(self, filename, exit_code, message, decryption=True):
        if exit_code in SOPS_ERROR_CODES:
            exception_name = SOPS_ERROR_CODES[exit_code]
            message = 'error with file %s: %s exited with code %d: %s' % (filename, exception_name, exit_code, to_native(message))
        else:
            message = 'could not %s file %s; Unknown sops error code: %s; message: %s' % ('decrypt' if decryption else 'encrypt', filename, exit_code, to_native(message))
        super(SopsError, self).__init__(message)