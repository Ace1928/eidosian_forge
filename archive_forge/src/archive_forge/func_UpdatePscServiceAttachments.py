from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdatePscServiceAttachments(unused_instance_ref, args, patch_request):
    """Hook to update psc confing service attachments to the update mask of the request."""
    if args.IsSpecified('psc_service_attachment'):
        _WarnForPscServiceAttachmentsUpdate()
        patch_request = AddFieldToUpdateMask('psc_config.service_attachments', patch_request)
    elif args.IsSpecified('clear_psc_service_attachments'):
        _WarnForPscServiceAttachmentsRemovalUpdate()
        patch_request = AddFieldToUpdateMask('psc_config.service_attachments', patch_request)
    return patch_request