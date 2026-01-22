from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateStringField(arg_name):
    """Validates that the string field is not longer than STRING_MAX_LENGTH, to avoid abuse issues."""
    if len(arg_name) > STRING_MAX_LENGTH:
        raise exceptions.BadArgumentException(arg_name, 'The string field can not be longer than {0} characters.'.format(STRING_MAX_LENGTH))
    return arg_name