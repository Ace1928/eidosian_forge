from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def SplitRequirementSpecifier(requirement_specifier):
    """Splits the package name from the other components of a requirement spec.

  Only supports PEP 508 `name_req` requirement specifiers. Does not support
  requirement specifiers containing environment markers.

  Args:
    requirement_specifier: str, a PEP 508 requirement specifier that does not
      contain an environment marker.

  Returns:
    (string, string), a 2-tuple of the extracted package name and the tail of
    the requirement specifier which could contain extras and/or a version
    specifier.

  Raises:
    Error: No package name was found in the requirement spec.
  """
    package = requirement_specifier.strip()
    tail_start_regex = '(\\[|\\(|==|>=|!=|<=|<|>|~=|===)'
    tail_match = re.search(tail_start_regex, requirement_specifier)
    tail = ''
    if tail_match:
        package = requirement_specifier[:tail_match.start()].strip()
        tail = requirement_specifier[tail_match.start():].strip()
    if not package:
        raise Error("Missing package name in requirement specifier: \\'{}\\'".format(requirement_specifier))
    return (package, tail)