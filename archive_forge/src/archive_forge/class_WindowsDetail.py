from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsDetail(_messages.Message):
    """A WindowsDetail object.

  Fields:
    cpeUri: Required. The [CPE URI](https://cpe.mitre.org/specification/) this
      vulnerability affects.
    description: The description of this vulnerability.
    fixingKbs: Required. The names of the KBs which have hotfixes to mitigate
      this vulnerability. Note that there may be multiple hotfixes (and thus
      multiple KBs) that mitigate a given vulnerability. Currently any listed
      KBs presence is considered a fix.
    name: Required. The name of this vulnerability.
  """
    cpeUri = _messages.StringField(1)
    description = _messages.StringField(2)
    fixingKbs = _messages.MessageField('KnowledgeBase', 3, repeated=True)
    name = _messages.StringField(4)