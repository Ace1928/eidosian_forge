import re
import sys
def _MSVSOnly(tool, name, setting_type):
    """Defines a setting that is only found in MSVS.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    name: the name of the setting.
    setting_type: the type of this setting.
  """

    def _Translate(unused_value, unused_msbuild_settings):
        pass
    _msvs_validators[tool.msvs_name][name] = setting_type.ValidateMSVS
    _msvs_to_msbuild_converters[tool.msvs_name][name] = _Translate