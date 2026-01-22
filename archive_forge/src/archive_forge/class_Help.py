from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Help(_messages.Message):
    """Provides links to documentation or for performing an out of band action.
  For example, if a quota check failed with an error indicating the calling
  project hasn't enabled the accessed service, this can contain a URL pointing
  directly to the right place in the developer console to flip the bit.

  Fields:
    links: URL(s) pointing to additional information on handling the current
      error.
  """
    links = _messages.MessageField('HelpLink', 1, repeated=True)