from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
class MemberParseError(exceptions.Error):
    """Error if a member is not in correct format."""