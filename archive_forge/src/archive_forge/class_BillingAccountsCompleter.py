from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
class BillingAccountsCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(BillingAccountsCompleter, self).__init__(collection='cloudbilling.billingAccounts', list_command='billing accounts list --uri', **kwargs)