from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
def GetResources(self, args, errors):
    health_checks = super(List, self).GetResources(args, errors)
    protocol_value = None
    if args.protocol is not None:
        protocol_value = self._ConvertProtocolArgToValue(args)
        if protocol_value not in self._ProtocolAllowlist():
            raise exceptions.ArgumentError('Invalid health check protocol ' + args.protocol + '.')
    for health_check in health_checks:
        if protocol_value is None or health_check['type'] == args.protocol.upper():
            yield health_check