from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def SingularizeCommandAndGroupDocumentID(name):
    """Returns singlularized name if name is 'COMMANDS' or 'GROUPS'."""
    return re.sub('(COMMAND|GROUP)S$', '\\1', name)