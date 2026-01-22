from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _RemoveComputeSection(existing_content):
    """Returns the contents of ssh_config_file with compute section removed.

  Args:
   existing_content: str, Current content of ssh config file.

  Raises:
    MultipleComputeSectionsError: If the config file has multiple compute
      sections already.

  Returns:
    A string containing the contents of ssh_config_file with the compute
    section removed.
  """
    match = _SECTION_RE.search(existing_content)
    if not match:
        return existing_content
    if not _ComputeSectionValid(existing_content):
        raise MultipleComputeSectionsError()
    return existing_content[:match.start(1)] + existing_content[match.end(1):]