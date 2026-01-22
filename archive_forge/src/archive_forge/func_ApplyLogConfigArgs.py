from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ApplyLogConfigArgs(messages, args, backend_service, cleared_fields=None):
    """Applies the LogConfig arguments to the specified backend service.

  If there are no arguments related to LogConfig, the backend service
  remains unmodified.

  Args:
    messages: The available API proto messages.
    args: The arguments passed to the gcloud command.
    backend_service: The backend service proto message object.
    cleared_fields: Reference to list with fields that should be cleared. Valid
      only for update command.
  """
    logging_specified = args.IsSpecified('enable_logging') or args.IsSpecified('logging_sample_rate') or args.IsSpecified('logging_optional') or args.IsSpecified('logging_optional_fields')
    valid_protocols = [messages.BackendService.ProtocolValueValuesEnum.HTTP, messages.BackendService.ProtocolValueValuesEnum.HTTPS, messages.BackendService.ProtocolValueValuesEnum.HTTP2, messages.BackendService.ProtocolValueValuesEnum.TCP, messages.BackendService.ProtocolValueValuesEnum.SSL, messages.BackendService.ProtocolValueValuesEnum.UDP, messages.BackendService.ProtocolValueValuesEnum.UNSPECIFIED]
    if logging_specified and backend_service.protocol not in valid_protocols:
        raise exceptions.InvalidArgumentException('--protocol', 'can only specify --enable-logging, --logging-sample-rate, --logging-optional or --logging-optional-fields if the protocol is HTTP/HTTPS/HTTP2/TCP/SSL/UDP/UNSPECIFIED.')
    if logging_specified:
        if backend_service.logConfig:
            log_config = backend_service.logConfig
        else:
            log_config = messages.BackendServiceLogConfig()
        if args.enable_logging is not None:
            log_config.enable = args.enable_logging
        if args.logging_sample_rate is not None:
            log_config.sampleRate = args.logging_sample_rate
        if args.logging_optional is not None:
            log_config.optionalMode = messages.BackendServiceLogConfig.OptionalModeValueValuesEnum(args.logging_optional)
        if args.logging_optional_fields is not None:
            log_config.optionalFields = args.logging_optional_fields
            if not args.logging_optional_fields and cleared_fields is not None:
                cleared_fields.append('logConfig.optionalFields')
        backend_service.logConfig = log_config