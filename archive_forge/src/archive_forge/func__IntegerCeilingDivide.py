from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _IntegerCeilingDivide(numerator, denominator):
    """returns numerator/denominator rounded up if there is any remainder."""
    return -(-numerator // denominator)