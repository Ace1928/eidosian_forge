from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.meta import cache_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _RequireConfirmation(name):
    """Prompt for cache deletion and return confirmation."""
    console_io.PromptContinue(message='The entire [{}] cache will be deleted.'.format(name), cancel_on_no=True, default=True)