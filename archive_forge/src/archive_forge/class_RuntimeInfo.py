from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeInfo(_messages.Message):
    """Runtime information about workload execution.

  Messages:
    EndpointsValue: Output only. Map of remote access endpoints (such as web
      interfaces and APIs) to their URIs.

  Fields:
    approximateUsage: Output only. Approximate workload resource usage,
      calculated when the workload completes (see Dataproc Serverless pricing
      (https://cloud.google.com/dataproc-serverless/pricing)).Note: This
      metric calculation may change in the future, for example, to capture
      cumulative workload resource consumption during workload execution (see
      the Dataproc Serverless release notes
      (https://cloud.google.com/dataproc-serverless/docs/release-notes) for
      announcements, changes, fixes and other Dataproc developments).
    currentUsage: Output only. Snapshot of current workload resource usage.
    diagnosticOutputUri: Output only. A URI pointing to the location of the
      diagnostics tarball.
    endpoints: Output only. Map of remote access endpoints (such as web
      interfaces and APIs) to their URIs.
    outputUri: Output only. A URI pointing to the location of the stdout and
      stderr of the workload.
    propertiesInfo: Optional. Properties of the workload organized by origin.
    publicKeys: Output only. The public keys used for sharing data with this
      workload.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EndpointsValue(_messages.Message):
        """Output only. Map of remote access endpoints (such as web interfaces
    and APIs) to their URIs.

    Messages:
      AdditionalProperty: An additional property for a EndpointsValue object.

    Fields:
      additionalProperties: Additional properties of type EndpointsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EndpointsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    approximateUsage = _messages.MessageField('UsageMetrics', 1)
    currentUsage = _messages.MessageField('UsageSnapshot', 2)
    diagnosticOutputUri = _messages.StringField(3)
    endpoints = _messages.MessageField('EndpointsValue', 4)
    outputUri = _messages.StringField(5)
    propertiesInfo = _messages.MessageField('PropertiesInfo', 6)
    publicKeys = _messages.MessageField('PublicKeys', 7)