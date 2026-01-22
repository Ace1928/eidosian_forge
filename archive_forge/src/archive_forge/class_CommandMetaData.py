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
class CommandMetaData(object):
    """Object containing metadata of command to be passed into linter renderer."""

    def __init__(self, flags=None, bool_flags=None, is_group=True):
        self.flags = flags if flags else []
        self.bool_flags = bool_flags if bool_flags else []
        self.is_group = is_group