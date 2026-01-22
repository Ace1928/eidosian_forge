from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _MetadataMessageToDict(metadata_message):
    """Converts a Metadata message to a dict."""
    res = {}
    if metadata_message:
        for item in metadata_message.items:
            res[item.key] = item.value
    return res