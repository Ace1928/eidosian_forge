from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.core.console import console_io
def ConfirmationPrompt(kind, items, verb):
    title = 'The following {} will be {}.'.format(kind, verb)
    console_io.PromptContinue(message=gke_util.ConstructList(title, items), throw_if_unattended=True, cancel_on_no=True)