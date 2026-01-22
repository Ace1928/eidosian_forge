import re
import sys
def _Moved(tool, settings_name, msbuild_tool_name, setting_type):
    _MovedAndRenamed(tool, settings_name, msbuild_tool_name, settings_name, setting_type)