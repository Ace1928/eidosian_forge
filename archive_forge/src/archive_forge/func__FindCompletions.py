from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def _FindCompletions(root, cmd_line):
    """Try to perform a completion based on the static CLI tree.

  Args:
    root: The root of the tree that will be traversed to find completions.
    cmd_line: [str], original command line.

  Raises:
    CannotHandleCompletionError: If FindCompletions cannot handle completion.

  Returns:
    []: No completions.
    [completions]: List, all possible sorted completions.
  """
    words = _GetCmdWordQueue(cmd_line)
    node = root
    global_flags = node[LOOKUP_FLAGS]
    completions = []
    flag_mode = FLAG_BOOLEAN
    env_var_prefix = GetEnvVarPrefix()
    env_vars = os.environ
    while words:
        word = words.pop()
        if word.startswith(FLAG_PREFIX):
            is_flag_word = True
            child_nodes = node.get(LOOKUP_FLAGS, {})
            child_nodes.update(global_flags)
            if _VALUE_SEP in word:
                word, flag_value = word.split(_VALUE_SEP, 1)
                words.append(flag_value)
        elif word.startswith(env_var_prefix):
            is_flag_word = False
            child_nodes = env_vars
            flag_mode = ENV_VAR
        else:
            is_flag_word = False
            child_nodes = node.get(LOOKUP_COMMANDS, {})
        if words:
            if word in child_nodes:
                if is_flag_word:
                    flag_mode = child_nodes[word]
                else:
                    flag_mode = FLAG_BOOLEAN
                    node = child_nodes[word]
            elif flag_mode == ENV_VAR:
                continue
            elif flag_mode != FLAG_BOOLEAN:
                flag_mode = FLAG_BOOLEAN
                continue
            elif not is_flag_word and (not node.get(LOOKUP_COMMANDS)):
                flag_mode = FLAG_BOOLEAN
                continue
            else:
                return []
        elif flag_mode == FLAG_DYNAMIC:
            raise CannotHandleCompletionError('Dynamic completions are not handled by this module')
        elif flag_mode == FLAG_VALUE:
            return []
        elif flag_mode == ENV_VAR:
            completions += MatchEnvVars(word, child_nodes)
        elif flag_mode != FLAG_BOOLEAN:
            for value in flag_mode:
                if value.startswith(word):
                    completions.append(value)
        elif not child_nodes:
            raise CannotHandleCompletionError('Positional completions are not handled by this module')
        else:
            for child, value in six.iteritems(child_nodes):
                if not child.startswith(word):
                    continue
                if is_flag_word and value != FLAG_BOOLEAN:
                    child += _VALUE_SEP
                completions.append(child)
    return sorted(completions)