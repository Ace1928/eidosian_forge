import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def ExpandXcodeVariables(string, expansions):
    """Expands Xcode-style $(VARIABLES) in string per the expansions dict.

  In some rare cases, it is appropriate to expand Xcode variables when a
  project file is generated.  For any substring $(VAR) in string, if VAR is a
  key in the expansions dict, $(VAR) will be replaced with expansions[VAR].
  Any $(VAR) substring in string for which VAR is not a key in the expansions
  dict will remain in the returned string.
  """
    matches = _xcode_variable_re.findall(string)
    if matches is None:
        return string
    matches.reverse()
    for match in matches:
        to_replace, variable = match
        if variable not in expansions:
            continue
        replacement = expansions[variable]
        string = re.sub(re.escape(to_replace), replacement, string)
    return string