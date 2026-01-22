from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
class Indent(object):
    """Second line indent stack."""

    def __init__(self):
        self.indent = TextRenderer.INDENT
        self.second_line_indent = self.indent