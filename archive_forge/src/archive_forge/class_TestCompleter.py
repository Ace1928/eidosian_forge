from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
class TestCompleter(ListCommandCompleter):
    """A completer that checks env var _ARGCOMPLETE_TEST for completer info.

  For testing list command completers.

  The env var is a comma separated list of name=value items:
    collection=COLLECTION
      The collection name.
    list_command=COMMAND
      The gcloud list command string with gcloud omitted.
  """

    def __init__(self, **kwargs):
        test_parameters = encoding.GetEncodedValue(os.environ, '_ARGCOMPLETE_TEST', 'parameters=bad')
        kwargs = dict(kwargs)
        for pair in test_parameters.split(','):
            name, value = pair.split('=')
            kwargs[name] = value
        if 'collection' not in kwargs or 'list_command' not in kwargs:
            raise TestParametersRequired('Specify test completer parameters in the _ARGCOMPLETE_TEST environment variable. It is a comma-separated separated list of name=value test parameters and must contain at least "collection=COLLECTION,list_command=LIST COMMAND" parameters.')
        super(TestCompleter, self).__init__(**kwargs)