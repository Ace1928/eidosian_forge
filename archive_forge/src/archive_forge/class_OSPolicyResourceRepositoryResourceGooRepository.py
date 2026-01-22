from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceRepositoryResourceGooRepository(_messages.Message):
    """Represents a Goo package repository. These are added to a repo file that
  is managed at `C:/ProgramData/GooGet/repos/google_osconfig.repo`.

  Fields:
    name: Required. The name of the repository.
    url: Required. The url of the repository.
  """
    name = _messages.StringField(1)
    url = _messages.StringField(2)