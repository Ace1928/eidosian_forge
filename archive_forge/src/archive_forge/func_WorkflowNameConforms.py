from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
def WorkflowNameConforms(name):
    """Confirm workflow name is of acceptable length and uses valid characters."""
    if not 1 <= len(name) <= 64:
        raise exceptions.InvalidArgumentException('workflow', 'ID must be between 1-64 characters long')
    if not re.search('^[a-zA-Z].*', name):
        raise exceptions.InvalidArgumentException('workflow', 'ID must start with a letter')
    if not re.search('.*[a-zA-Z0-9]$', name):
        raise exceptions.InvalidArgumentException('workflow', 'ID must end with a letter or number')
    if not re.search('^[-_a-zA-Z0-9]*$', name):
        raise exceptions.InvalidArgumentException('workflow', 'ID must only contain letters, numbers, underscores and hyphens')