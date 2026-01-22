import re
import sys
def _Same(tool, name, setting_type):
    """Defines a setting that has the same name in MSVS and MSBuild.

  Args:
    tool: a dictionary that gives the names of the tool for MSVS and MSBuild.
    name: the name of the setting.
    setting_type: the type of this setting.
  """
    _Renamed(tool, name, name, setting_type)