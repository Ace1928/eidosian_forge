from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
def ParseFirewallActions(actions):
    messages = apis.GetMessagesModule('recaptchaenterprise', 'v1')
    actions_list = actions.split(',')
    action_messages = []
    for action in actions_list:
        action_messages.append(ParseAction(action, messages))
    return action_messages