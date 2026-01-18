from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
List Insight Types.

    Args:
      page_size: int, The number of items to retrieve per request.
      limit: int, The maximum number of records to yield.

    Returns:
      The list of insight types.
    