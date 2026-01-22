from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _GetTagAndStatus(tag):
    """Returns the tag with padding and an adding/removing indicator if needed."""
    if tag.inSpec and (not tag.inStatus):
        return '  {} (Adding):'.format(tag.tag)
    elif not tag.inSpec and tag.inStatus:
        return '  {} (Deleting):'.format(tag.tag)
    else:
        return '  {}:'.format(tag.tag)