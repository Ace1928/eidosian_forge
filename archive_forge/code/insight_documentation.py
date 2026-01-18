from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
Mark an insight's state as DISMISSED.

    Args:
      name: str, the name of the insight being updated.
      etag: Fingerprint of the Insight. Provides optimistic locking when
        updating states.

    Returns:
      The result insights after being marked as dismissed
    