from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible.module_utils.common.validation import (
def txt_character_encoding(self):
    """
        Return how the API handles escape sequences in TXT records.

        Returns one of the following strings:
        * 'octal' - the API works with octal escape sequences
        * 'decimal' - the API works with decimal escape sequences

        This return value is only used if txt_record_handling returns 'encoded'.

        WARNING: the default return value will change to 'decimal' for community.dns 3.0.0!
        """
    return 'octal'