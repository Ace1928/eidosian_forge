from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
def ProcessPfxFile(domain_ref, args, request):
    """Reads the pfx file into the LDAPSSettings proto and updates the request."""
    if args.certificate_pfx_file:
        if not request.lDAPSSettings:
            messages = util.GetMessagesForResource(domain_ref)
            settings = messages.LDAPSSettings()
            request.lDAPSSettings = settings
        request.lDAPSSettings.certificatePfx = args.certificate_pfx_file
        request = AddFieldToUpdateMask('certificatePfx', request)
        request = AddFieldToUpdateMask('certificatePassword', request)
    return request