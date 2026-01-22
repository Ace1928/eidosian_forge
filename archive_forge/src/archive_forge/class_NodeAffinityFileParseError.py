from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
class NodeAffinityFileParseError(Error):
    """Exception for invalid node affinity file format."""