from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class UnsupportedNameException(Error):
    """Exception raised if a name is incompatible with Graphviz ID escaping."""