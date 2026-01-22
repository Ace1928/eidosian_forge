from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
class ComponentVersion(object):
    """Version information for the component.

  Attributes:
    build_number: int, The unique, monotonically increasing version of the
      component.
    version_string: str, The user facing version for the component.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('build_number', required=True)
        p.Parse('version_string', required=True)
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('build_number')
        w.Write('version_string')
        return w.Dictionary()

    def __init__(self, build_number, version_string):
        self.build_number = build_number
        self.version_string = version_string