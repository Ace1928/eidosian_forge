from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _PickSymbol(best, alt, encoding):
    """Chooses the best symbol (if it's in this encoding) or an alternate.

  Args:
    best: str, the symbol to return if the encoding allows.
    alt: str, alternative to return if best cannot be encoded.
    encoding:  str, possible values are utf8, ascii, and win.

  Returns:
    The symbol string if the encoding allows, otherwise an alternative string.
  """
    try:
        best.encode(encoding)
        return best
    except UnicodeError:
        return alt