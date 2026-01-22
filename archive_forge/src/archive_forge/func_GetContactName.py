from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetContactName(args):
    """Returns the full contact name from the args or raises an exception."""
    c = GetContactId(args)
    p = GetParent(args)
    return '{}/contacts/{}'.format(p, c)