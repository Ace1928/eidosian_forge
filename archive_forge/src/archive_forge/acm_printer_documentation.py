from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import difflib
import io
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import yaml_printer
Tweak yaml printer formatted resources for ACM's dry run describe output.

    Args:
      lines: yaml printer formatted strings

    Returns:
      lines with no '-' prefix for yaml array elements.
    