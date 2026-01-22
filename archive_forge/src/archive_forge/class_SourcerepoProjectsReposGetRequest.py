from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposGetRequest(_messages.Message):
    """A SourcerepoProjectsReposGetRequest object.

  Fields:
    name: The name of the requested repository. Values are of the form
      `projects//repos/`.
  """
    name = _messages.StringField(1, required=True)