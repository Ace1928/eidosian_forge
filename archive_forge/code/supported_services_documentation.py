from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.accesscontextmanager import util
Make API call to list VPC Service Controls supported services.

    Args:
      page_size: The page size to list.
      limit: The maximum number of services to display.

    Returns:
      The list of VPC Service Controls supported services
    