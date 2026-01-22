from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
def Success(msg, **formatters):
    """Print a nice little green checkbox, and a message.

  Args:
    msg: str, message accepting standard formatters.
    **formatters: extra args acting like .format()
  """
    _Print('✓ ', 'green', msg, **formatters)