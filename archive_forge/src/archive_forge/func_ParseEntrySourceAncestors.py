from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from typing import List
from googlecloudsdk.api_lib.dataplex import util as dataplex_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
def ParseEntrySourceAncestors(ancestors: List[str]):
    """Parse ancestors from a string.

  Args:
    ancestors: A list of strings containing the JSON representation of the
      Ancestors.

  Returns:
    A list of ancestors parsed to a proto message
    (GoogleCloudDataplexV1EntrySourceAncestor).
  """
    if ancestors is None:
        return []
    return list(map(lambda ancestor: messages_util.DictToMessageWithErrorCheck(json.loads(ancestor), dataplex_message.GoogleCloudDataplexV1EntrySourceAncestor), ancestors))