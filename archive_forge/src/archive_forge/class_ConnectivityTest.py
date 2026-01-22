from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectivityTest(_messages.Message):
    """A Connectivity Test for a network reachability analysis.

  Messages:
    LabelsValue: Resource labels to represent user-provided metadata.

  Fields:
    bypassFirewallChecks: Whether the test should skip firewall checking. If
      not provided, we assume false.
    createTime: Output only. The time the test was created.
    description: The user-supplied description of the Connectivity Test.
      Maximum of 512 characters.
    destination: Required. Destination specification of the Connectivity Test.
      You can use a combination of destination IP address, Compute Engine VM
      instance, or VPC network to uniquely identify the destination location.
      Even if the destination IP address is not unique, the source IP location
      is unique. Usually, the analysis can infer the destination endpoint from
      route information. If the destination you specify is a VM instance and
      the instance has multiple network interfaces, then you must also specify
      either a destination IP address or VPC network to identify the
      destination interface. A reachability analysis proceeds even if the
      destination location is ambiguous. However, the result can include
      endpoints that you don't intend to test.
    displayName: Output only. The display name of a Connectivity Test.
    labels: Resource labels to represent user-provided metadata.
    name: Required. Unique name of the resource using the form:
      `projects/{project_id}/locations/global/connectivityTests/{test_id}`
    probingDetails: Output only. The probing details of this test from the
      latest run, present for applicable tests only. The details are updated
      when creating a new test, updating an existing test, or triggering a
      one-time rerun of an existing test.
    protocol: IP Protocol of the test. When not provided, "TCP" is assumed.
    reachabilityDetails: Output only. The reachability details of this test
      from the latest run. The details are updated when creating a new test,
      updating an existing test, or triggering a one-time rerun of an existing
      test.
    relatedProjects: Other projects that may be relevant for reachability
      analysis. This is applicable to scenarios where a test can cross project
      boundaries.
    source: Required. Source specification of the Connectivity Test. You can
      use a combination of source IP address, virtual machine (VM) instance,
      or Compute Engine network to uniquely identify the source location.
      Examples: If the source IP address is an internal IP address within a
      Google Cloud Virtual Private Cloud (VPC) network, then you must also
      specify the VPC network. Otherwise, specify the VM instance, which
      already contains its internal IP address and VPC network information. If
      the source of the test is within an on-premises network, then you must
      provide the destination VPC network. If the source endpoint is a Compute
      Engine VM instance with multiple network interfaces, the instance itself
      is not sufficient to identify the endpoint. So, you must also specify
      the source IP address or VPC network. A reachability analysis proceeds
      even if the source location is ambiguous. However, the test result may
      include endpoints that you don't intend to test.
    updateTime: Output only. The time the test's configuration was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Resource labels to represent user-provided metadata.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bypassFirewallChecks = _messages.BooleanField(1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    destination = _messages.MessageField('Endpoint', 4)
    displayName = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    probingDetails = _messages.MessageField('ProbingDetails', 8)
    protocol = _messages.StringField(9)
    reachabilityDetails = _messages.MessageField('ReachabilityDetails', 10)
    relatedProjects = _messages.StringField(11, repeated=True)
    source = _messages.MessageField('Endpoint', 12)
    updateTime = _messages.StringField(13)