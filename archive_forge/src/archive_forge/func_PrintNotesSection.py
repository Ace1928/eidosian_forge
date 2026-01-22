from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def PrintNotesSection(self, disable_header=False):
    """Prints the NOTES section if needed.

    Args:
      disable_header: Disable printing the section header if True.
    """
    notes = self.GetNotes()
    if notes:
        if not disable_header:
            self.PrintSectionHeader('NOTES')
        if notes:
            self._out(notes + '\n\n')