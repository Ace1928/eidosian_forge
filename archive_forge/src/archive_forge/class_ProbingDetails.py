from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProbingDetails(_messages.Message):
    """Results of active probing from the last run of the test.

  Enums:
    AbortCauseValueValuesEnum: The reason probing was aborted.
    ResultValueValuesEnum: The overall result of active probing.

  Fields:
    abortCause: The reason probing was aborted.
    destinationEgressLocation: The EdgeLocation from which a packet destined
      for/originating from the internet will egress/ingress the Google
      network. This will only be populated for a connectivity test which has
      an internet destination/source address. The absence of this field *must
      not* be used as an indication that the destination/source is part of the
      Google network.
    endpointInfo: The source and destination endpoints derived from the test
      input and used for active probing.
    error: Details about an internal failure or the cancellation of active
      probing.
    probingLatency: Latency as measured by active probing in one direction:
      from the source to the destination endpoint.
    result: The overall result of active probing.
    sentProbeCount: Number of probes sent.
    successfulProbeCount: Number of probes that reached the destination.
    verifyTime: The time that reachability was assessed through active
      probing.
  """

    class AbortCauseValueValuesEnum(_messages.Enum):
        """The reason probing was aborted.

    Values:
      PROBING_ABORT_CAUSE_UNSPECIFIED: No reason was specified.
      PERMISSION_DENIED: The user lacks permission to access some of the
        network resources required to run the test.
      NO_SOURCE_LOCATION: No valid source endpoint could be derived from the
        request.
    """
        PROBING_ABORT_CAUSE_UNSPECIFIED = 0
        PERMISSION_DENIED = 1
        NO_SOURCE_LOCATION = 2

    class ResultValueValuesEnum(_messages.Enum):
        """The overall result of active probing.

    Values:
      PROBING_RESULT_UNSPECIFIED: No result was specified.
      REACHABLE: At least 95% of packets reached the destination.
      UNREACHABLE: No packets reached the destination.
      REACHABILITY_INCONSISTENT: Less than 95% of packets reached the
        destination.
      UNDETERMINED: Reachability could not be determined. Possible reasons
        are: * The user lacks permission to access some of the network
        resources required to run the test. * No valid source endpoint could
        be derived from the request. * An internal error occurred.
    """
        PROBING_RESULT_UNSPECIFIED = 0
        REACHABLE = 1
        UNREACHABLE = 2
        REACHABILITY_INCONSISTENT = 3
        UNDETERMINED = 4
    abortCause = _messages.EnumField('AbortCauseValueValuesEnum', 1)
    destinationEgressLocation = _messages.MessageField('EdgeLocation', 2)
    endpointInfo = _messages.MessageField('EndpointInfo', 3)
    error = _messages.MessageField('Status', 4)
    probingLatency = _messages.MessageField('LatencyDistribution', 5)
    result = _messages.EnumField('ResultValueValuesEnum', 6)
    sentProbeCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    successfulProbeCount = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    verifyTime = _messages.StringField(9)