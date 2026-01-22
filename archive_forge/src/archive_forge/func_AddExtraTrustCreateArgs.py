from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def AddExtraTrustCreateArgs(unused_ref, args, request):
    """Allows for the handshake secret to be read from stdin if not specified."""
    if args.IsSpecified('handshake_secret'):
        return request
    secret = GetHandshakeSecret()
    request.attachTrustRequest.trust.trustHandshakeSecret = secret
    return request