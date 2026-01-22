import re
import sys
def _MSBuildOnly(tool, name, setting_type):
    """Defines a setting that is only found in MSBuild.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    name: the name of the setting.
    setting_type: the type of this setting.
  """

    def _Translate(value, msbuild_settings):
        tool_settings = msbuild_settings.setdefault(tool.msbuild_name, {})
        tool_settings[name] = value
    _msbuild_validators[tool.msbuild_name][name] = setting_type.ValidateMSBuild
    _msvs_to_msbuild_converters[tool.msvs_name][name] = _Translate