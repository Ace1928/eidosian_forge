from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import re
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def GetCommandSuggestions(command_words):
    """Return suggested commands containing input command words.

  Args:
    command_words: List of input command words.

  Returns:
    [command]: A list of canonical command strings with 'gcloud' prepended. Only
      commands whose scores have a ratio of at least MIN_RATIO against the top
      score are returned. At most MAX_SUGGESTIONS command strings are returned.
      If many commands from the same group are being suggested, then the common
      groups are suggested instead.
  """
    suggested_commands = []
    try:
        scored_commands = _GetScoredCommandsContaining(command_words)
    except lookup.CannotHandleCompletionError:
        scored_commands = None
    if not scored_commands:
        return suggested_commands
    top_score = float(scored_commands[0][1])
    too_many = False
    suggested_groups = set()
    for command, score in scored_commands:
        if score / top_score >= MIN_RATIO:
            suggested_commands.append(' '.join(['gcloud'] + command))
            suggested_groups.add(' '.join(command[:-1]))
            if len(suggested_commands) >= MAX_SUGGESTIONS:
                too_many = True
                break
    if too_many and len(suggested_groups) < MIN_SUGGESTED_GROUPS:
        min_length = len(scored_commands[0][0])
        for command, score in scored_commands:
            if score / top_score < MIN_RATIO:
                break
            if min_length > len(command):
                min_length = len(command)
        common_length = min_length - 1
        if common_length:
            suggested_groups = set()
            for command, score in scored_commands:
                if score / top_score < MIN_RATIO:
                    break
                suggested_groups.add(' '.join(['gcloud'] + command[:common_length]))
                if len(suggested_groups) >= MAX_SUGGESTIONS:
                    break
            suggested_commands = sorted(suggested_groups)
    return suggested_commands