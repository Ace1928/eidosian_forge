from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPythonPackagesResponse(_messages.Message):
    """The response from listing python packages.

  Fields:
    nextPageToken: The token to retrieve the next page of artifacts, or empty
      if there are no more artifacts to return.
    pythonPackages: The python packages returned.
  """
    nextPageToken = _messages.StringField(1)
    pythonPackages = _messages.MessageField('PythonPackage', 2, repeated=True)