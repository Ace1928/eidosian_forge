from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def RaiseOnPromptFailure():
    """Call this to raise an exn when prompt cannot read from input stream."""
    phrases = ('one of ', 'flags') if len(flag_names) > 1 else ('', 'flag')
    raise compute_exceptions.FailedPromptError('Unable to prompt. Specify {0}the [{1}] {2}.'.format(phrases[0], ', '.join(flag_names), phrases[1]))