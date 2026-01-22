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
def _WordScore(index, normalized_command_word, canonical_command_word, canonical_command_length):
    """Returns the integer word match score for a command word.

  Args:
    index: The position of the word in the command.
    normalized_command_word: The normalized command word.
    canonical_command_word: The actual command word to compare with.
    canonical_command_length: The length of the actual command.

  Returns:
    The integer word match score, always >= 0.
  """
    score = 0
    if normalized_command_word in canonical_command_word:
        shorter_word = normalized_command_word
        longer_word = canonical_command_word
    elif canonical_command_word in normalized_command_word:
        shorter_word = canonical_command_word
        longer_word = normalized_command_word
    else:
        return score
    hit = longer_word.find(shorter_word)
    if hit > 0 and longer_word[hit - 1] != '-':
        return score
    score += 10
    if canonical_command_length == 1:
        score += 30
    elif canonical_command_length == 2:
        score += 20
    elif canonical_command_length == 3:
        score += 10
    if index == 0:
        score += 25
    elif index == 1:
        score += 15
    else:
        score += 5
    extra = len(longer_word) - len(shorter_word)
    if extra <= 2:
        extra = 3 - extra
        if longer_word.startswith(shorter_word):
            extra *= 2
        score += extra
    if index == 0 and canonical_command_length > 1:
        score += 30
    elif index > 0 and canonical_command_length > index + 1:
        score += 15
    return score