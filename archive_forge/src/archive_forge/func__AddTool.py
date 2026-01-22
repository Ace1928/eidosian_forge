import re
import sys
def _AddTool(tool):
    """Adds a tool to the four dictionaries used to process settings.

  This only defines the tool.  Each setting also needs to be added.

  Args:
    tool: The _Tool object to be added.
  """
    _msvs_validators[tool.msvs_name] = {}
    _msbuild_validators[tool.msbuild_name] = {}
    _msvs_to_msbuild_converters[tool.msvs_name] = {}
    _msbuild_name_of_tool[tool.msvs_name] = tool.msbuild_name