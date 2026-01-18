from __future__ import unicode_literals
from six import string_types
from prompt_toolkit.completion import Completer, Completion
def word_matches(word):
    """ True when the word before the cursor matches. """
    if self.ignore_case:
        word = word.lower()
    if self.match_middle:
        return word_before_cursor in word
    else:
        return word.startswith(word_before_cursor)