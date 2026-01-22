from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from typing import Any
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags
import six
class FunctionResourceCommand(six.with_metaclass(abc.ABCMeta, base.Command)):
    """Mix-in for single function resource commands that work with both v1 or v2.

  Which version of the command to run is determined by the following precedence:
  1. Explicit setting via the --gen2/--no-gen2 flags or functions/gen2 property.
  2. The generation of the function if it exists.
  2. The v1 API by default in GA, the v2 API in Beta/Alpha.

  Subclasses should add the function resource arg and --gen2 flag.
  """

    def __init__(self, *args, **kwargs):
        super(FunctionResourceCommand, self).__init__(*args, **kwargs)
        self._v2_function = None

    @abc.abstractmethod
    def _RunV1(self, args: parser_extensions.Namespace) -> Any:
        """Runs the command against the v1 API."""

    @abc.abstractmethod
    def _RunV2(self, args: parser_extensions.Namespace) -> Any:
        """Runs the command against the v2 API."""

    @api_util_v1.CatchHTTPErrorRaiseHTTPException
    def Run(self, args: parser_extensions.Namespace) -> Any:
        """Runs the command.

    Args:
      args: The arguments this command was invoked with.

    Returns:
      The result of the command.

    Raises:
      HttpException: If an HttpError occurs.
    """
        if flags.ShouldUseGen2():
            return self._RunV2(args)
        if flags.ShouldUseGen1():
            return self._RunV1(args)
        client = client_v2.FunctionsClient(self.ReleaseTrack())
        self._v2_function = client.GetFunction(args.CONCEPTS.name.Parse().RelativeName())
        if self._v2_function:
            if str(self._v2_function.environment) == 'GEN_2':
                return self._RunV2(args)
            else:
                return self._RunV1(args)
        if self.ReleaseTrack() == base.ReleaseTrack.GA:
            return self._RunV1(args)
        return self._RunV2(args)