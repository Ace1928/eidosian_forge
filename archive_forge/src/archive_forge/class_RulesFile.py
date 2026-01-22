from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RulesFile(_messages.Message):
    """Details of a single rules file.

  Fields:
    rulesContent: Required. The text content of the rules that needs to be
      converted.
    rulesSourceFilename: Required. The filename of the rules that needs to be
      converted. The filename is used mainly so that future logs of the import
      rules job contain it, and can therefore be searched by it.
  """
    rulesContent = _messages.StringField(1)
    rulesSourceFilename = _messages.StringField(2)