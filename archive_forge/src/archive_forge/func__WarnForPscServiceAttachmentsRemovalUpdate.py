from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def _WarnForPscServiceAttachmentsRemovalUpdate():
    """Adds prompt that warns about service attachments removal."""
    message = 'Removal of instance PSC service attachments requested. '
    console_io.PromptContinue(message=message, prompt_string='Do you want to proceed with removal of service attachments?', cancel_on_no=True)