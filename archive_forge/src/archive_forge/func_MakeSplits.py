from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def MakeSplits(split_list):
    """Convert a string list to a Split object.

  Args:
    split_list: A list that contains strings representing splitting points.

  Returns:
    A Split object.
  """
    results = []
    for split in split_list:
        results.append(util.GetAdminMessages().Split(key=split.encode('utf-8')))
    return results