from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GooRepository(_messages.Message):
    """Represents a Goo package repository. These is added to a repo file that
  is stored at C:/ProgramData/GooGet/repos/google_osconfig.repo.

  Fields:
    name: Required. The name of the repository.
    url: Required. The url of the repository.
  """
    name = _messages.StringField(1)
    url = _messages.StringField(2)