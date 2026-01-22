from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def GetCustomerId(args):
    """Return customer_id.

  Args:
    args: The argparse namespace.

  Returns:
    customer_id.

  """
    if hasattr(args, 'customer') and args.IsSpecified('customer'):
        customer_id = args.customer
    elif hasattr(args, 'organization') and args.IsSpecified('organization'):
        customer_id = ConvertOrgArgToObfuscatedCustomerId(args.organization)
    return 'customerId/' + customer_id