from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def AddAuthFieldMask(unused_ref, args, request):
    """Adds auth-specific fieldmask entries."""
    if args.auth_type is None:
        return request
    if request.updateMask:
        request.updateMask += ',authentication'
    else:
        request.updateMask = 'authentication'
    return request