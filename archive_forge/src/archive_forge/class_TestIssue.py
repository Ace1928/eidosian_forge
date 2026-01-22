from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestIssue(_messages.Message):
    """An issue detected occurring during a test execution.

  Enums:
    CategoryValueValuesEnum: Category of issue. Required.
    SeverityValueValuesEnum: Severity of issue. Required.
    TypeValueValuesEnum: Type of issue. Required.

  Fields:
    category: Category of issue. Required.
    errorMessage: A brief human-readable message describing the issue.
      Required.
    severity: Severity of issue. Required.
    stackTrace: Deprecated in favor of stack trace fields inside specific
      warnings.
    type: Type of issue. Required.
    warning: Warning message with additional details of the issue. Should
      always be a message from com.google.devtools.toolresults.v1.warnings
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Category of issue. Required.

    Values:
      unspecifiedCategory: Default unspecified category. Do not use. For
        versioning only.
      common: Issue is not specific to a particular test kind (e.g., a native
        crash).
      robo: Issue is specific to Robo run.
    """
        unspecifiedCategory = 0
        common = 1
        robo = 2

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity of issue. Required.

    Values:
      unspecifiedSeverity: Default unspecified severity. Do not use. For
        versioning only.
      info: Non critical issue, providing users with some info about the test
        run.
      suggestion: Non critical issue, providing users with some hints on
        improving their testing experience, e.g., suggesting to use Game
        Loops.
      warning: Potentially critical issue.
      severe: Critical issue.
    """
        unspecifiedSeverity = 0
        info = 1
        suggestion = 2
        warning = 3
        severe = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Type of issue. Required.

    Values:
      unspecifiedType: Default unspecified type. Do not use. For versioning
        only.
      fatalException: Issue is a fatal exception.
      nativeCrash: Issue is a native crash.
      anr: Issue is an ANR crash.
      unusedRoboDirective: Issue is an unused robo directive.
      compatibleWithOrchestrator: Issue is a suggestion to use orchestrator.
      launcherActivityNotFound: Issue with finding a launcher activity
      startActivityNotFound: Issue with resolving a user-provided intent to
        start an activity
      incompleteRoboScriptExecution: A Robo script was not fully executed.
      completeRoboScriptExecution: A Robo script was fully and successfully
        executed.
      failedToInstall: The APK failed to install.
      availableDeepLinks: The app-under-test has deep links, but none were
        provided to Robo.
      nonSdkApiUsageViolation: App accessed a non-sdk Api.
      nonSdkApiUsageReport: App accessed a non-sdk Api (new detailed report)
      encounteredNonAndroidUiWidgetScreen: Robo crawl encountered at least one
        screen with elements that are not Android UI widgets.
      encounteredLoginScreen: Robo crawl encountered at least one probable
        login screen.
      performedGoogleLogin: Robo signed in with Google.
      iosException: iOS App crashed with an exception.
      iosCrash: iOS App crashed without an exception (e.g. killed).
      performedMonkeyActions: Robo crawl involved performing some monkey
        actions.
      usedRoboDirective: Robo crawl used a Robo directive.
      usedRoboIgnoreDirective: Robo crawl used a Robo directive to ignore an
        UI element.
      insufficientCoverage: Robo did not crawl some potentially important
        parts of the app.
      inAppPurchases: Robo crawl involved some in-app purchases.
      crashDialogError: Crash dialog was detected during the test execution
      uiElementsTooDeep: UI element depth is greater than the threshold
      blankScreen: Blank screen is found in the Robo crawl
      overlappingUiElements: Overlapping UI elements are found in the Robo
        crawl
      unityException: An uncaught Unity exception was detected (these don't
        crash apps).
      deviceOutOfMemory: Device running out of memory was detected
      logcatCollectionError: Problems detected while collecting logcat
      detectedAppSplashScreen: Robo detected a splash screen provided by app
        (vs. Android OS splash screen).
      assetIssue: There was an issue with the assets in this test.
    """
        unspecifiedType = 0
        fatalException = 1
        nativeCrash = 2
        anr = 3
        unusedRoboDirective = 4
        compatibleWithOrchestrator = 5
        launcherActivityNotFound = 6
        startActivityNotFound = 7
        incompleteRoboScriptExecution = 8
        completeRoboScriptExecution = 9
        failedToInstall = 10
        availableDeepLinks = 11
        nonSdkApiUsageViolation = 12
        nonSdkApiUsageReport = 13
        encounteredNonAndroidUiWidgetScreen = 14
        encounteredLoginScreen = 15
        performedGoogleLogin = 16
        iosException = 17
        iosCrash = 18
        performedMonkeyActions = 19
        usedRoboDirective = 20
        usedRoboIgnoreDirective = 21
        insufficientCoverage = 22
        inAppPurchases = 23
        crashDialogError = 24
        uiElementsTooDeep = 25
        blankScreen = 26
        overlappingUiElements = 27
        unityException = 28
        deviceOutOfMemory = 29
        logcatCollectionError = 30
        detectedAppSplashScreen = 31
        assetIssue = 32
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    errorMessage = _messages.StringField(2)
    severity = _messages.EnumField('SeverityValueValuesEnum', 3)
    stackTrace = _messages.MessageField('StackTrace', 4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)
    warning = _messages.MessageField('Any', 6)