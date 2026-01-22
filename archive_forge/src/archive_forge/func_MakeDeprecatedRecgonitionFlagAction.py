from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def MakeDeprecatedRecgonitionFlagAction(flag_name):
    return actions.DeprecationAction('--' + flag_name, warn='The `{}` flag is deprecated and will be removed. The Google Cloud Speech-to-text api does not use it, and only passes it through back into response.'.format(flag_name))