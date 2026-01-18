from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
Visit each command and group in the CLI command tree.

      Args:
        command: dict, The tree (nested dict) of command/group names.
        prefix: [str], The subcommand arg prefix.
      