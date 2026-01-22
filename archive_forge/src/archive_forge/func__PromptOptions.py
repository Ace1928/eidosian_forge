from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def _PromptOptions(options, type_):
    """Given an iterable options of type type_, prompt and return one."""
    options = sorted(set(options), key=str)
    if len(options) > 1:
        idx = console_io.PromptChoice(options, message='Which {0}?'.format(type_))
    elif len(options) == 1:
        idx = 0
        log.status.Print('Choosing [{0}] for {1}.\n'.format(options[0], type_))
    else:
        if all_instances:
            msg = 'No instances could be found matching the given criteria.\n\nAll instances:\n' + '\n'.join(map('* [{0}]'.format, sorted(all_instances, key=str)))
        else:
            msg = 'No instances were found for the current project [{0}].'.format(properties.VALUES.core.project.Get(required=True))
        raise SelectInstanceError(msg)
    return options[idx]