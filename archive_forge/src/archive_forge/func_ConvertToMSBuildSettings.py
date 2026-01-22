import re
import sys
def ConvertToMSBuildSettings(msvs_settings, stderr=sys.stderr):
    """Converts MSVS settings (VS2008 and earlier) to MSBuild settings (VS2010+).

  Args:
      msvs_settings: A dictionary.  The key is the tool name.  The values are
          themselves dictionaries of settings and their values.
      stderr: The stream receiving the error messages.

  Returns:
      A dictionary of MSBuild settings.  The key is either the MSBuild tool name
      or the empty string (for the global settings).  The values are themselves
      dictionaries of settings and their values.
  """
    msbuild_settings = {}
    for msvs_tool_name, msvs_tool_settings in msvs_settings.items():
        if msvs_tool_name in _msvs_to_msbuild_converters:
            msvs_tool = _msvs_to_msbuild_converters[msvs_tool_name]
            for msvs_setting, msvs_value in msvs_tool_settings.items():
                if msvs_setting in msvs_tool:
                    try:
                        msvs_tool[msvs_setting](msvs_value, msbuild_settings)
                    except ValueError as e:
                        print('Warning: while converting %s/%s to MSBuild, %s' % (msvs_tool_name, msvs_setting, e), file=stderr)
                else:
                    _ValidateExclusionSetting(msvs_setting, msvs_tool, 'Warning: unrecognized setting %s/%s while converting to MSBuild.' % (msvs_tool_name, msvs_setting), stderr)
        else:
            print('Warning: unrecognized tool %s while converting to MSBuild.' % msvs_tool_name, file=stderr)
    return msbuild_settings