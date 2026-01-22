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
class MultipleComputeSectionsError(exceptions.ToolException):
    """Multiple compute sections are disallowed."""

    def __init__(self, ssh_config_file='the SSH configuration file'):
        super(MultipleComputeSectionsError, self).__init__('Found more than one Google Compute Engine section in [{0}]. You can either delete [{0}] and let this command recreate it for you or you can manually delete all sections marked with [{1}] and [{2}].'.format(ssh_config_file, _BEGIN_MARKER, _END_MARKER))