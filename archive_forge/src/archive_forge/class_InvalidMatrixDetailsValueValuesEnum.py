from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InvalidMatrixDetailsValueValuesEnum(_messages.Enum):
    """Output only. Describes why the matrix is considered invalid. Only
    useful for matrices in the INVALID state.

    Values:
      INVALID_MATRIX_DETAILS_UNSPECIFIED: Do not use. For proto versioning
        only.
      DETAILS_UNAVAILABLE: The matrix is INVALID, but there are no further
        details available.
      MALFORMED_APK: The input app APK could not be parsed.
      MALFORMED_TEST_APK: The input test APK could not be parsed.
      NO_MANIFEST: The AndroidManifest.xml could not be found.
      NO_PACKAGE_NAME: The APK manifest does not declare a package name.
      INVALID_PACKAGE_NAME: The APK application ID (aka package name) is
        invalid. See also
        https://developer.android.com/studio/build/application-id
      TEST_SAME_AS_APP: The test package and app package are the same.
      NO_INSTRUMENTATION: The test apk does not declare an instrumentation.
      NO_SIGNATURE: The input app apk does not have a signature.
      INSTRUMENTATION_ORCHESTRATOR_INCOMPATIBLE: The test runner class
        specified by user or in the test APK's manifest file is not compatible
        with Android Test Orchestrator. Orchestrator is only compatible with
        AndroidJUnitRunner version 1.1 or higher. Orchestrator can be disabled
        by using DO_NOT_USE_ORCHESTRATOR OrchestratorOption.
      NO_TEST_RUNNER_CLASS: The test APK does not contain the test runner
        class specified by the user or in the manifest file. This can be
        caused by one of the following reasons: - the user provided a runner
        class name that's incorrect, or - the test runner isn't built into the
        test APK (might be in the app APK instead).
      NO_LAUNCHER_ACTIVITY: A main launcher activity could not be found.
      FORBIDDEN_PERMISSIONS: The app declares one or more permissions that are
        not allowed.
      INVALID_ROBO_DIRECTIVES: There is a conflict in the provided
        robo_directives.
      INVALID_RESOURCE_NAME: There is at least one invalid resource name in
        the provided robo directives
      INVALID_DIRECTIVE_ACTION: Invalid definition of action in the robo
        directives (e.g. a click or ignore action includes an input text
        field)
      TEST_LOOP_INTENT_FILTER_NOT_FOUND: There is no test loop intent filter,
        or the one that is given is not formatted correctly.
      SCENARIO_LABEL_NOT_DECLARED: The request contains a scenario label that
        was not declared in the manifest.
      SCENARIO_LABEL_MALFORMED: There was an error when parsing a label's
        value.
      SCENARIO_NOT_DECLARED: The request contains a scenario number that was
        not declared in the manifest.
      DEVICE_ADMIN_RECEIVER: Device administrator applications are not
        allowed.
      MALFORMED_XC_TEST_ZIP: The zipped XCTest was malformed. The zip did not
        contain a single .xctestrun file and the contents of the
        DerivedData/Build/Products directory.
      BUILT_FOR_IOS_SIMULATOR: The zipped XCTest was built for the iOS
        simulator rather than for a physical device.
      NO_TESTS_IN_XC_TEST_ZIP: The .xctestrun file did not specify any test
        targets.
      USE_DESTINATION_ARTIFACTS: One or more of the test targets defined in
        the .xctestrun file specifies "UseDestinationArtifacts", which is
        disallowed.
      TEST_NOT_APP_HOSTED: XC tests which run on physical devices must have
        "IsAppHostedTestBundle" == "true" in the xctestrun file.
      PLIST_CANNOT_BE_PARSED: An Info.plist file in the XCTest zip could not
        be parsed.
      TEST_ONLY_APK: The APK is marked as "testOnly". Deprecated and not
        currently used.
      MALFORMED_IPA: The input IPA could not be parsed.
      MISSING_URL_SCHEME: The application doesn't register the game loop URL
        scheme.
      MALFORMED_APP_BUNDLE: The iOS application bundle (.app) couldn't be
        processed.
      NO_CODE_APK: APK contains no code. See also
        https://developer.android.com/guide/topics/manifest/application-
        element.html#code
      INVALID_INPUT_APK: Either the provided input APK path was malformed, the
        APK file does not exist, or the user does not have permission to
        access the APK file.
      INVALID_APK_PREVIEW_SDK: APK is built for a preview SDK which is
        unsupported
      MATRIX_TOO_LARGE: The matrix expanded to contain too many executions.
      TEST_QUOTA_EXCEEDED: Not enough test quota to run the executions in this
        matrix.
      SERVICE_NOT_ACTIVATED: A required cloud service api is not activated.
        See: https://firebase.google.com/docs/test-
        lab/android/continuous#requirements
      UNKNOWN_PERMISSION_ERROR: There was an unknown permission issue running
        this test.
    """
    INVALID_MATRIX_DETAILS_UNSPECIFIED = 0
    DETAILS_UNAVAILABLE = 1
    MALFORMED_APK = 2
    MALFORMED_TEST_APK = 3
    NO_MANIFEST = 4
    NO_PACKAGE_NAME = 5
    INVALID_PACKAGE_NAME = 6
    TEST_SAME_AS_APP = 7
    NO_INSTRUMENTATION = 8
    NO_SIGNATURE = 9
    INSTRUMENTATION_ORCHESTRATOR_INCOMPATIBLE = 10
    NO_TEST_RUNNER_CLASS = 11
    NO_LAUNCHER_ACTIVITY = 12
    FORBIDDEN_PERMISSIONS = 13
    INVALID_ROBO_DIRECTIVES = 14
    INVALID_RESOURCE_NAME = 15
    INVALID_DIRECTIVE_ACTION = 16
    TEST_LOOP_INTENT_FILTER_NOT_FOUND = 17
    SCENARIO_LABEL_NOT_DECLARED = 18
    SCENARIO_LABEL_MALFORMED = 19
    SCENARIO_NOT_DECLARED = 20
    DEVICE_ADMIN_RECEIVER = 21
    MALFORMED_XC_TEST_ZIP = 22
    BUILT_FOR_IOS_SIMULATOR = 23
    NO_TESTS_IN_XC_TEST_ZIP = 24
    USE_DESTINATION_ARTIFACTS = 25
    TEST_NOT_APP_HOSTED = 26
    PLIST_CANNOT_BE_PARSED = 27
    TEST_ONLY_APK = 28
    MALFORMED_IPA = 29
    MISSING_URL_SCHEME = 30
    MALFORMED_APP_BUNDLE = 31
    NO_CODE_APK = 32
    INVALID_INPUT_APK = 33
    INVALID_APK_PREVIEW_SDK = 34
    MATRIX_TOO_LARGE = 35
    TEST_QUOTA_EXCEEDED = 36
    SERVICE_NOT_ACTIVATED = 37
    UNKNOWN_PERMISSION_ERROR = 38