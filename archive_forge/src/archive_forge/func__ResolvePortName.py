from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import log
def _ResolvePortName(args):
    """Determine port name if one was not specified."""
    if args.port_name:
        return args.port_name

    def _LogAndReturn(port_name):
        log.status.Print("Backend-services' port_name automatically resolved to {} based on the service protocol.".format(port_name))
        return port_name
    if args.protocol == 'HTTPS':
        return _LogAndReturn('https')
    if args.protocol == 'HTTP2':
        return _LogAndReturn('http2')
    if args.protocol == 'SSL':
        return _LogAndReturn('ssl')
    if args.protocol == 'TCP':
        return _LogAndReturn('tcp')
    return 'http'