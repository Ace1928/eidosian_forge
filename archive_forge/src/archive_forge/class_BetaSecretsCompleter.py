from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
class BetaSecretsCompleter(SecretsCompleter):

    def __init__(self, **kwargs):
        super(BetaSecretsCompleter, self).__init__(list_command='beta secrets list --uri', **kwargs)