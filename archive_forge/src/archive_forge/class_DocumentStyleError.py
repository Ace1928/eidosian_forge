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
class DocumentStyleError(exceptions.Error):
    """An exception for unknown document styles."""

    def __init__(self, style):
        message = 'Unknown markdown document style [{style}] -- must be one of: {styles}.'.format(style=style, styles=', '.join(sorted(STYLES.keys())))
        super(DocumentStyleError, self).__init__(message)