from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
def _CompleteFromCompleterClass(self, prefix='', cache=None, parsed_args=None):
    """Helper to complete from a class."""
    if parsed_args and len(parsed_args._GetCommand().ai.positional_completers) > 1:
        qualified_parameter_names = {'collection'}
    else:
        qualified_parameter_names = set()
    completer = None
    try:
        completer = self._completer_class(cache=cache, qualified_parameter_names=qualified_parameter_names)
        parameter_info = completer.ParameterInfo(parsed_args, self._argument)
        return completer.Complete(prefix, parameter_info)
    except BaseException as e:
        if isinstance(e, TypeError) and (not completer):
            return self._CompleteFromGenericCompleterClass(prefix=prefix)
        return self._HandleCompleterException(e, prefix=prefix, completer=completer)