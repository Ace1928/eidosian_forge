from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import re
from typing import Any, Optional
from apitools.base.py import encoding
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
def _PrintAsYaml(self, content: Any) -> str:
    buffer = io.StringIO()
    printer = yp.YamlPrinter(buffer)
    printer.Print(content)
    return buffer.getvalue()