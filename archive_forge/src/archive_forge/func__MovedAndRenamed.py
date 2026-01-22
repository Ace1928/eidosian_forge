import re
import sys
def _MovedAndRenamed(tool, msvs_settings_name, msbuild_tool_name, msbuild_settings_name, setting_type):
    """Defines a setting that may have moved to a new section.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    msvs_settings_name: the MSVS name of the setting.
    msbuild_tool_name: the name of the MSBuild tool to place the setting under.
    msbuild_settings_name: the MSBuild name of the setting.
    setting_type: the type of this setting.
  """

    def _Translate(value, msbuild_settings):
        tool_settings = msbuild_settings.setdefault(msbuild_tool_name, {})
        tool_settings[msbuild_settings_name] = setting_type.ConvertToMSBuild(value)
    _msvs_validators[tool.msvs_name][msvs_settings_name] = setting_type.ValidateMSVS
    validator = setting_type.ValidateMSBuild
    _msbuild_validators[msbuild_tool_name][msbuild_settings_name] = validator
    _msvs_to_msbuild_converters[tool.msvs_name][msvs_settings_name] = _Translate