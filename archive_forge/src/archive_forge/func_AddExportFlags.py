from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import os
import re
import textwrap
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding as api_encoding
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import encoding
def AddExportFlags(parser, schema_path=None):
    """Add common export flags to the arg parser.

  Args:
    parser: The argparse parser object.
    schema_path: The resource instance schema file path if there is one.
  """
    help_text = 'Path to a YAML file where the configuration will be exported.\n          Alternatively, you may omit this flag to write to standard output.'
    if schema_path is not None:
        help_text += ' For a schema describing the export/import format, see:\n          {}.\n      '.format(schema_path)
    parser.add_argument('--destination', help=textwrap.dedent(help_text), required=False)