from __future__ import unicode_literals
class Lint(object):
    """Basically a named-tuple to store a lint id, message format, and metadata
  """

    def __init__(self, idstr, msgfmt, description=None, explain=None):
        self.idstr = idstr
        self.msgfmt = msgfmt
        self.description = description
        self.explain = explain