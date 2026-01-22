from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def PathTransform(r):
    """A resource transform to get the command path with search terms stylized.

  Uses the "results" attribute of the command to determine which terms to
  stylize and the "path" attribute of the command to get the command path.

  Args:
    r: a json representation of a command.

  Returns:
    str, the path of the command with search terms stylized.
  """
    results = r[lookup.RESULTS]
    path = ' '.join(r[lookup.PATH])
    return Highlight(path, results.keys())