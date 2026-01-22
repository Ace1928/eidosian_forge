from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ParseMoney(money):
    """Parses money string as tuple (units, cents, currency)."""
    match = re.match('^(\\d+|\\d+\\.\\d{2})\\s*([A-Z]{3})$', money)
    if match:
        number, s = match.groups()
    else:
        raise ValueError('Value could not be parsed as number + currency code')
    if '.' in number:
        index = number.find('.')
        return (int(number[:index]), int(number[index + 1:]), s)
    else:
        return (int(number), 0, s)