from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _ExpirationPolicy(self, no_expiration, expiration_period):
    """Build ExpirationPolicy message from argument values.

    Args:
      no_expiration (bool): Whether or not to set no expiration on subscription.
      expiration_period (str): TTL on expiration_policy.

    Returns:
      ExpirationPolicy message or None.
    """
    if no_expiration:
        return self.messages.ExpirationPolicy(ttl=None)
    if expiration_period:
        return self.messages.ExpirationPolicy(ttl=expiration_period)
    return None