from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
Make API calls to List active Cloud NetApp operations.

    Args:
      location_ref: The parsed location of the listed NetApp resources.
      limit: The number of Cloud NetApp resources to limit the results to. This
        limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud NetApp resources.
    