from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _NormalizeFolder(folder):
    """Validates and normalizes a folder name."""
    if '/' in folder:
        if _FOLDER_NAME_PATTERN.fullmatch(folder):
            return folder + _PARENT_SUFFIX
        raise _InvalidFullResourcePathForPattern(_FOLDER_NAME_PATTERN)
    return 'folders/' + folder + _PARENT_SUFFIX