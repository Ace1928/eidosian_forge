from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.document_renderers import devsite_scripts
from googlecloudsdk.core.document_renderers import html_renderer
def FlushExample(self):
    """Prints full example string with devsite tags to output stream."""
    self._out.write('<code class="devsite-terminal">')
    self._out.write(self.WrapFlags('span', '-(-\\w+)+', ['flag']))
    self._out.write('</code>\n')
    self._whole_example = ''