from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import resources
class IamServiceAccountCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(IamServiceAccountCompleter, self).__init__(list_command='iam service-accounts list --quiet --flatten=email --format=disable', **kwargs)