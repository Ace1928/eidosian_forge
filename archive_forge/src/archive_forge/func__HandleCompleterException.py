from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
def _HandleCompleterException(self, exception, prefix, completer=None):
    """Handles completer errors by crafting two "completions" from exception.

    Fatal completer errors return two "completions", each an error
    message that is displayed by the shell completers, and look more
    like a pair of error messages than completions. This is much better than
    the default that falls back to the file completer with no indication of
    errors, typically yielding the list of all files in the current directory.

    NOTICE: Each message must start with different characters, otherwise they
    will be taken as valid completions. Also, the messages are sorted in the
    display, so the messages here are displayed with ERROR first and REASON
    second.

    Args:
      exception: The completer exception.
      prefix: The current prefix string to be matched by the completer.
      completer: The instantiated completer object or None.

    Returns:
      Two "completions" crafted from the completer exception.
    """
    if completer and hasattr(completer, 'collection'):
        completer_name = completer.collection
    else:
        completer_name = self._completer_class.__name__
    return self._MakeCompletionErrorMessages(['{}ERROR: {} resource completer failed.'.format(prefix, completer_name), '{}REASON: {}'.format(prefix, six.text_type(exception))])