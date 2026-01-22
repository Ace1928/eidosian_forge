from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
import sys
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.document_renderers import devsite_renderer
from googlecloudsdk.core.document_renderers import html_renderer
from googlecloudsdk.core.document_renderers import linter_renderer
from googlecloudsdk.core.document_renderers import man_renderer
from googlecloudsdk.core.document_renderers import markdown_renderer
from googlecloudsdk.core.document_renderers import renderer
from googlecloudsdk.core.document_renderers import text_renderer
def _Finish(self):
    """Flushes the fill buffer and checks for NOTES.

    A previous _ConvertHeading() will have cleared self._notes if a NOTES
    section has already been seen.

    Returns:
      The renderer Finish() value.
    """
    self._Fill()
    if self._notes:
        self._renderer.Line()
        self._renderer.Heading(2, 'NOTES')
        self._buf += self._notes
        self._Fill()
    return self._renderer.Finish()