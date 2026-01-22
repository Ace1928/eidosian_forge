from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def _WarnForPscServiceAttachmentsUpdate():
    """Adds prompt that warns about service attachments update."""
    message = 'Change to instance PSC service attachments requested. '
    message += 'Updating the PSC service attachments from cli means the value provided will be considered as the entire list and not an amendment to the existing list of PSC service attachments'
    console_io.PromptContinue(message=message, prompt_string='Do you want to proceed with update?', cancel_on_no=True)