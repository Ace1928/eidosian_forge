from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedService(_messages.Message):
    """`SupportedService` specifies the VPC Service Controls and its
  properties.

  Enums:
    SupportStageValueValuesEnum: The support stage of the service.

  Fields:
    availableOnRestrictedVip: True if the service is available on the
      restricted VIP. Services on the restricted VIP typically either support
      VPC Service Controls or are core infrastructure services required for
      the functioning of Google Cloud.
    knownLimitations: True if the service is supported with some limitations.
      Check [documentation](https://cloud.google.com/vpc-service-
      controls/docs/supported-products) for details.
    name: The service name/address of the supported service, such as
      'service.googleapis.com'
    supportStage: The support stage of the service.
    supportedMethods: The list of the supported methods. This field exists
      only in response to GetSupportedService
    title: The name of the supported product, such as 'Cloud Product API'.
  """

    class SupportStageValueValuesEnum(_messages.Enum):
        """The support stage of the service.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: Do not use this default value.
      UNIMPLEMENTED: The feature is not yet implemented. Users can not use it.
      PRELAUNCH: Prelaunch features are hidden from users and are only visible
        internally.
      EARLY_ACCESS: Early Access features are limited to a closed group of
        testers. To use these features, you must sign up in advance and sign a
        Trusted Tester agreement (which includes confidentiality provisions).
        These features may be unstable, changed in backward-incompatible ways,
        and are not guaranteed to be released.
      ALPHA: Alpha is a limited availability test for releases before they are
        cleared for widespread use. By Alpha, all significant design issues
        are resolved and we are in the process of verifying functionality.
        Alpha customers need to apply for access, agree to applicable terms,
        and have their projects allowlisted. Alpha releases don't have to be
        feature complete, no SLAs are provided, and there are no technical
        support obligations, but they will be far enough along that customers
        can actually use them in test environments or for limited-use tests --
        just like they would in normal production cases.
      BETA: Beta is the point at which we are ready to open a release for any
        customer to use. There are no SLA or technical support obligations in
        a Beta release. Products will be complete from a feature perspective,
        but may have some open outstanding issues. Beta releases are suitable
        for limited production use cases.
      GA: GA features are open to all developers and are considered stable and
        fully qualified for production use.
      DEPRECATED: Deprecated features are scheduled to be shut down and
        removed. For more information, see the "Deprecation Policy" section of
        our [Terms of Service](https://cloud.google.com/terms/) and the
        [Google Cloud Platform Subject to the Deprecation
        Policy](https://cloud.google.com/terms/deprecation) documentation.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        UNIMPLEMENTED = 1
        PRELAUNCH = 2
        EARLY_ACCESS = 3
        ALPHA = 4
        BETA = 5
        GA = 6
        DEPRECATED = 7
    availableOnRestrictedVip = _messages.BooleanField(1)
    knownLimitations = _messages.BooleanField(2)
    name = _messages.StringField(3)
    supportStage = _messages.EnumField('SupportStageValueValuesEnum', 4)
    supportedMethods = _messages.MessageField('MethodSelector', 5, repeated=True)
    title = _messages.StringField(6)