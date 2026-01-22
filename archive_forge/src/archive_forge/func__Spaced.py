from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_property
def _Spaced(lines):
    """Adds a line of space between the passed in lines."""
    spaced_lines = []
    for line in lines:
        if spaced_lines:
            spaced_lines.append(' ')
        spaced_lines.append(line)
    return spaced_lines