from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientLibrarySettings(_messages.Message):
    """Details about how and where to publish client libraries.

  Enums:
    LaunchStageValueValuesEnum: Launch stage of this version of the API.

  Fields:
    cppSettings: Settings for C++ client libraries.
    dotnetSettings: Settings for .NET client libraries.
    goSettings: Settings for Go client libraries.
    javaSettings: Settings for legacy Java features, supported in the Service
      YAML.
    launchStage: Launch stage of this version of the API.
    nodeSettings: Settings for Node client libraries.
    phpSettings: Settings for PHP client libraries.
    pythonSettings: Settings for Python client libraries.
    restNumericEnums: When using transport=rest, the client request will
      encode enums as numbers rather than strings.
    rubySettings: Settings for Ruby client libraries.
    version: Version of the API to apply these settings to. This is the full
      protobuf package for the API, ending in the version element. Examples:
      "google.cloud.speech.v1" and "google.spanner.admin.database.v1".
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """Launch stage of this version of the API.

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
    cppSettings = _messages.MessageField('CppSettings', 1)
    dotnetSettings = _messages.MessageField('DotnetSettings', 2)
    goSettings = _messages.MessageField('GoSettings', 3)
    javaSettings = _messages.MessageField('JavaSettings', 4)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 5)
    nodeSettings = _messages.MessageField('NodeSettings', 6)
    phpSettings = _messages.MessageField('PhpSettings', 7)
    pythonSettings = _messages.MessageField('PythonSettings', 8)
    restNumericEnums = _messages.BooleanField(9)
    rubySettings = _messages.MessageField('RubySettings', 10)
    version = _messages.StringField(11)