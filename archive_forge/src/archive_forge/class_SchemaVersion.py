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
class SchemaVersion(object):
    """Information about the schema version of this snapshot file.

  Attributes:
    version: int, The schema version number.  A different number is considered
      incompatible.
    no_update: bool, True if this installation should not attempted to be
      updated.
    message: str, A message to display to the user if they are updating to this
      new schema version.
    url: str, The URL to grab a fresh Cloud SDK bundle.
  """

    @classmethod
    def FromDictionary(cls, dictionary):
        p = DictionaryParser(cls, dictionary)
        p.Parse('version', required=True)
        p.Parse('no_update', default=False)
        p.Parse('message')
        p.Parse('url', required=True)
        return cls(**p.Args())

    def ToDictionary(self):
        w = DictionaryWriter(self)
        w.Write('version')
        w.Write('no_update')
        w.Write('message')
        w.Write('url')
        return w.Dictionary()

    def __init__(self, version, no_update, message, url):
        self.version = version
        self.no_update = no_update
        self.message = message
        self.url = url