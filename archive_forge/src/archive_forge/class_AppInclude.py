from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class AppInclude(validation.Validated):
    """Class representing the contents of an included `app.yaml` file.

  This class is used for both `builtins` and `includes` directives.
  """
    ATTRIBUTES = {BUILTINS: validation.Optional(validation.Repeated(BuiltinHandler)), INCLUDES: validation.Optional(validation.Type(list)), HANDLERS: validation.Optional(validation.Repeated(URLMap), default=[]), ADMIN_CONSOLE: validation.Optional(AdminConsole), MANUAL_SCALING: validation.Optional(ManualScaling), VM: validation.Optional(bool), VM_SETTINGS: validation.Optional(VmSettings), BETA_SETTINGS: validation.Optional(BetaSettings), ENV_VARIABLES: validation.Optional(EnvironmentVariables), BUILD_ENV_VARIABLES: validation.Optional(EnvironmentVariables), SKIP_FILES: validation.RegexStr(default=SKIP_NO_FILES)}

    @classmethod
    def MergeManualScaling(cls, appinclude_one, appinclude_two):
        """Takes the greater of `<manual_scaling.instances>` from the arguments.

    `appinclude_one` is mutated to be the merged result in this process.

    Also, this function must be updated if `ManualScaling` gets additional
    fields.

    Args:
      appinclude_one: The first object to merge. The object must have a
          `manual_scaling` field that contains a `ManualScaling()`.
      appinclude_two: The second object to merge. The object must have a
          `manual_scaling` field that contains a `ManualScaling()`.
    Returns:
      An object that is the result of merging
      `appinclude_one.manual_scaling.instances` and
      `appinclude_two.manual_scaling.instances`; this is returned as a revised
      `appinclude_one` object after the mutations are complete.
    """

        def _Instances(appinclude):
            """Determines the number of `manual_scaling.instances` sets.

      Args:
        appinclude: The include for which you want to determine the number of
            `manual_scaling.instances` sets.

      Returns:
        The number of instances as an integer. If the value of
        `manual_scaling.instances` evaluates to False (e.g. 0 or None), then
        return 0.
      """
            if appinclude.manual_scaling:
                if appinclude.manual_scaling.instances:
                    return int(appinclude.manual_scaling.instances)
            return 0
        if _Instances(appinclude_one) or _Instances(appinclude_two):
            instances = max(_Instances(appinclude_one), _Instances(appinclude_two))
            appinclude_one.manual_scaling = ManualScaling(instances=str(instances))
        return appinclude_one

    @classmethod
    def _CommonMergeOps(cls, one, two):
        """This function performs common merge operations.

    Args:
      one: The first object that you want to merge.
      two: The second object that you want to merge.

    Returns:
      An updated `one` object containing all merged data.
    """
        AppInclude.MergeManualScaling(one, two)
        one.admin_console = AdminConsole.Merge(one.admin_console, two.admin_console)
        one.vm = two.vm or one.vm
        one.vm_settings = VmSettings.Merge(one.vm_settings, two.vm_settings)
        if hasattr(one, 'beta_settings'):
            one.beta_settings = BetaSettings.Merge(one.beta_settings, two.beta_settings)
        one.env_variables = EnvironmentVariables.Merge(one.env_variables, two.env_variables)
        one.skip_files = cls.MergeSkipFiles(one.skip_files, two.skip_files)
        return one

    @classmethod
    def MergeAppYamlAppInclude(cls, appyaml, appinclude):
        """Merges an `app.yaml` file with referenced builtins/includes.

    Args:
      appyaml: The `app.yaml` file that you want to update with `appinclude`.
      appinclude: The includes that you want to merge into `appyaml`.

    Returns:
      An updated `app.yaml` file that includes the directives you specified in
      `appinclude`.
    """
        if not appinclude:
            return appyaml
        if appinclude.handlers:
            tail = appyaml.handlers or []
            appyaml.handlers = []
            for h in appinclude.handlers:
                if not h.position or h.position == 'head':
                    appyaml.handlers.append(h)
                else:
                    tail.append(h)
                h.position = None
            appyaml.handlers.extend(tail)
        appyaml = cls._CommonMergeOps(appyaml, appinclude)
        appyaml.NormalizeVmSettings()
        return appyaml

    @classmethod
    def MergeAppIncludes(cls, appinclude_one, appinclude_two):
        """Merges the non-referential state of the provided `AppInclude`.

    That is, `builtins` and `includes` directives are not preserved, but any
    static objects are copied into an aggregate `AppInclude` object that
    preserves the directives of both provided `AppInclude` objects.

    `appinclude_one` is updated to be the merged result in this process.

    Args:
      appinclude_one: First `AppInclude` to merge.
      appinclude_two: Second `AppInclude` to merge.

    Returns:
      `AppInclude` object that is the result of merging the static directives of
      `appinclude_one` and `appinclude_two`. An updated version of
      `appinclude_one` is returned.
    """
        if not appinclude_one or not appinclude_two:
            return appinclude_one or appinclude_two
        if appinclude_one.handlers:
            if appinclude_two.handlers:
                appinclude_one.handlers.extend(appinclude_two.handlers)
        else:
            appinclude_one.handlers = appinclude_two.handlers
        return cls._CommonMergeOps(appinclude_one, appinclude_two)

    @staticmethod
    def MergeSkipFiles(skip_files_one, skip_files_two):
        """Merges two `skip_files` directives.

    Args:
      skip_files_one: The first `skip_files` element that you want to merge.
      skip_files_two: The second `skip_files` element that you want to merge.

    Returns:
      A list of regular expressions that are merged.
    """
        if skip_files_one == SKIP_NO_FILES:
            return skip_files_two
        if skip_files_two == SKIP_NO_FILES:
            return skip_files_one
        return validation.RegexStr().Validate([skip_files_one, skip_files_two], SKIP_FILES)