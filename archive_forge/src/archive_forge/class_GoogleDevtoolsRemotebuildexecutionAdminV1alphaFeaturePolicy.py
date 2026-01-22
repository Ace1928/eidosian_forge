from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicy(_messages.Message):
    """FeaturePolicy defines features allowed to be used on RBE instances, as
  well as instance-wide behavior changes that take effect without opt-in or
  opt-out at usage time.

  Enums:
    ActionHermeticityValueValuesEnum: Defines the hermeticity policy for
      actions on this instance. DO NOT USE: Experimental / unlaunched feature.
    ActionIsolationValueValuesEnum: Defines the isolation policy for actions
      on this instance. DO NOT USE: Experimental / unlaunched feature.
    LinuxExecutionValueValuesEnum: Defines how Linux actions are allowed to
      execute. DO NOT USE: Experimental / unlaunched feature.
    LinuxIsolationValueValuesEnum: linux_isolation allows overriding the
      docker runtime used for containers started on Linux.
    MacExecutionValueValuesEnum: Defines how Windows actions are allowed to
      execute. DO NOT USE: Experimental / unlaunched feature.
    VmVerificationValueValuesEnum: Whether to verify CreateBotSession and
      UpdateBotSession from the bot.
    WindowsExecutionValueValuesEnum: Defines how Windows actions are allowed
      to execute. DO NOT USE: Experimental / unlaunched feature.

  Fields:
    actionHermeticity: Defines the hermeticity policy for actions on this
      instance. DO NOT USE: Experimental / unlaunched feature.
    actionIsolation: Defines the isolation policy for actions on this
      instance. DO NOT USE: Experimental / unlaunched feature.
    containerImageSources: Which container image sources are allowed.
      Currently only RBE-supported registry (gcr.io) is allowed. One can allow
      all repositories under a project or one specific repository only. E.g.
      container_image_sources { policy: RESTRICTED allowed_values: [
      "gcr.io/project-foo", "gcr.io/project-bar/repo-baz", ] } will allow any
      repositories under "gcr.io/project-foo" plus the repository
      "gcr.io/project-bar/repo-baz". Default (UNSPECIFIED) is equivalent to
      any source is allowed.
    dockerAddCapabilities: Whether dockerAddCapabilities can be used or what
      capabilities are allowed.
    dockerChrootPath: Whether dockerChrootPath can be used.
    dockerNetwork: Whether dockerNetwork can be used or what network modes are
      allowed. E.g. one may allow `off` value only via `allowed_values`.
    dockerPrivileged: Whether dockerPrivileged can be used.
    dockerRunAsContainerProvidedUser: Whether dockerRunAsContainerProvidedUser
      can be used.
    dockerRunAsRoot: Whether dockerRunAsRoot can be used.
    dockerRuntime: Whether dockerRuntime is allowed to be set or what runtimes
      are allowed. Note linux_isolation takes precedence, and if set,
      docker_runtime values may be rejected if they are incompatible with the
      selected isolation.
    dockerSiblingContainers: Whether dockerSiblingContainers can be used.
    dockerUlimits: Whether dockerUlimits are allowed to be set.
    linuxExecution: Defines how Linux actions are allowed to execute. DO NOT
      USE: Experimental / unlaunched feature.
    linuxIsolation: linux_isolation allows overriding the docker runtime used
      for containers started on Linux.
    macExecution: Defines how Windows actions are allowed to execute. DO NOT
      USE: Experimental / unlaunched feature.
    vmVerification: Whether to verify CreateBotSession and UpdateBotSession
      from the bot.
    windowsExecution: Defines how Windows actions are allowed to execute. DO
      NOT USE: Experimental / unlaunched feature.
  """

    class ActionHermeticityValueValuesEnum(_messages.Enum):
        """Defines the hermeticity policy for actions on this instance. DO NOT
    USE: Experimental / unlaunched feature.

    Values:
      ACTION_HERMETICITY_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to OFF.
      ACTION_HERMETICITY_OFF: Disables enforcing feature policies that
        guarantee action hermeticity.
      ACTION_HERMETICITY_ENFORCED: Enforces hermeticity of actions by
        requiring feature policies to be set that prevent actions from gaining
        network access. The enforcement mechanism has been reviewed by ISE.
      ACTION_HERMETICITY_BEST_EFFORT: Requires feature policies to be set that
        provide best effort hermeticity for actions. Best effort hermeticity
        means network access will be disabled and not trivial to bypass.
        However, a determined and malicious action may still find a way to
        gain network access.
    """
        ACTION_HERMETICITY_UNSPECIFIED = 0
        ACTION_HERMETICITY_OFF = 1
        ACTION_HERMETICITY_ENFORCED = 2
        ACTION_HERMETICITY_BEST_EFFORT = 3

    class ActionIsolationValueValuesEnum(_messages.Enum):
        """Defines the isolation policy for actions on this instance. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      ACTION_ISOLATION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to OFF.
      ACTION_ISOLATION_OFF: Disables enforcing feature policies that guarantee
        action isolation.
      ACTION_ISOLATION_ENFORCED: Enforces setting feature policies that
        ensures actions within the RBE Instance are isolated from each other
        in a way deemed sufficient by ISE reviewers.
    """
        ACTION_ISOLATION_UNSPECIFIED = 0
        ACTION_ISOLATION_OFF = 1
        ACTION_ISOLATION_ENFORCED = 2

    class LinuxExecutionValueValuesEnum(_messages.Enum):
        """Defines how Linux actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      LINUX_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      LINUX_EXECUTION_FORBIDDEN: Linux actions and worker pools are forbidden.
      LINUX_EXECUTION_UNRESTRICTED: No restrictions on execution of Linux
        actions.
      LINUX_EXECUTION_HARDENED_GVISOR: Linux actions will be hardened using
        gVisor. Actions that specify a configuration incompatible with gVisor
        hardening will be rejected. Example per-action platform properties
        that are incompatible with gVisor hardening are: 1. dockerRuntime is
        set to a value other than "runsc". Leaving dockerRuntime unspecified
        *is* compatible with gVisor. 2. dockerPrivileged is set to "true".
        etc.
      LINUX_EXECUTION_HARDENED_GVISOR_OR_TERMINAL: Linux actions will be
        hardened using gVisor if their configuration is compatible with gVisor
        hardening. Otherwise, the action will be terminal, i.e., the worker VM
        that runs the action will be terminated after the action finishes.
    """
        LINUX_EXECUTION_UNSPECIFIED = 0
        LINUX_EXECUTION_FORBIDDEN = 1
        LINUX_EXECUTION_UNRESTRICTED = 2
        LINUX_EXECUTION_HARDENED_GVISOR = 3
        LINUX_EXECUTION_HARDENED_GVISOR_OR_TERMINAL = 4

    class LinuxIsolationValueValuesEnum(_messages.Enum):
        """linux_isolation allows overriding the docker runtime used for
    containers started on Linux.

    Values:
      LINUX_ISOLATION_UNSPECIFIED: Default value. Will be using Linux default
        runtime.
      GVISOR: Use gVisor runsc runtime.
      OFF: Use stardard Linux runtime. This has the same behaviour as
        unspecified, but it can be used to revert back from gVisor.
    """
        LINUX_ISOLATION_UNSPECIFIED = 0
        GVISOR = 1
        OFF = 2

    class MacExecutionValueValuesEnum(_messages.Enum):
        """Defines how Windows actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      MAC_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      MAC_EXECUTION_FORBIDDEN: Mac actions and worker pools are forbidden.
      MAC_EXECUTION_UNRESTRICTED: No restrictions on execution of Mac actions.
    """
        MAC_EXECUTION_UNSPECIFIED = 0
        MAC_EXECUTION_FORBIDDEN = 1
        MAC_EXECUTION_UNRESTRICTED = 2

    class VmVerificationValueValuesEnum(_messages.Enum):
        """Whether to verify CreateBotSession and UpdateBotSession from the bot.

    Values:
      VM_VERIFICATION_UNSPECIFIED: Default value. Same as GCP_TOKEN.
      VM_VERIFICATION_GCP_TOKEN: Verify the VM token and the nonce associated
        with the VM.
      VM_VERIFICATION_OFF: Don't verify VM token and nonce.
    """
        VM_VERIFICATION_UNSPECIFIED = 0
        VM_VERIFICATION_GCP_TOKEN = 1
        VM_VERIFICATION_OFF = 2

    class WindowsExecutionValueValuesEnum(_messages.Enum):
        """Defines how Windows actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      WINDOWS_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      WINDOWS_EXECUTION_FORBIDDEN: Windows actions and worker pools are
        forbidden.
      WINDOWS_EXECUTION_UNRESTRICTED: No restrictions on execution of Windows
        actions.
      WINDOWS_EXECUTION_TERMINAL: Windows actions will always result in the
        worker VM being terminated after the action completes.
    """
        WINDOWS_EXECUTION_UNSPECIFIED = 0
        WINDOWS_EXECUTION_FORBIDDEN = 1
        WINDOWS_EXECUTION_UNRESTRICTED = 2
        WINDOWS_EXECUTION_TERMINAL = 3
    actionHermeticity = _messages.EnumField('ActionHermeticityValueValuesEnum', 1)
    actionIsolation = _messages.EnumField('ActionIsolationValueValuesEnum', 2)
    containerImageSources = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 3)
    dockerAddCapabilities = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 4)
    dockerChrootPath = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 5)
    dockerNetwork = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 6)
    dockerPrivileged = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 7)
    dockerRunAsContainerProvidedUser = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 8)
    dockerRunAsRoot = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 9)
    dockerRuntime = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 10)
    dockerSiblingContainers = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 11)
    dockerUlimits = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaFeaturePolicyFeature', 12)
    linuxExecution = _messages.EnumField('LinuxExecutionValueValuesEnum', 13)
    linuxIsolation = _messages.EnumField('LinuxIsolationValueValuesEnum', 14)
    macExecution = _messages.EnumField('MacExecutionValueValuesEnum', 15)
    vmVerification = _messages.EnumField('VmVerificationValueValuesEnum', 16)
    windowsExecution = _messages.EnumField('WindowsExecutionValueValuesEnum', 17)