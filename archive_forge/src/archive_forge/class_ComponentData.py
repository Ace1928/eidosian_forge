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
class ComponentData(object):
    """Information on the data source for the component.

  Attributes:
    type: str, The type of the source of this data (i.e. tar).
    source: str, The hosted location of the component.
    size: int, The size of the component in bytes.
    checksum: str, The hex digest of the archive file.
    contents_checksum: str, The hex digest of the contents of all files in the
      archive.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('type', required=True)
        p.Parse('source', required=True)
        p.Parse('size')
        p.Parse('checksum')
        p.Parse('contents_checksum')
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('type')
        w.Write('source')
        w.Write('size')
        w.Write('checksum')
        w.Write('contents_checksum')
        return w.Dictionary()

    def __init__(self, type, source, size, checksum, contents_checksum):
        self.type = type
        self.source = source
        self.size = size
        self.checksum = checksum
        self.contents_checksum = contents_checksum