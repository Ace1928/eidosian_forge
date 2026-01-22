from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsTestMatricesGetRequest(_messages.Message):
    """A TestingProjectsTestMatricesGetRequest object.

  Fields:
    projectId: Cloud project that owns the test matrix.
    testMatrixId: Unique test matrix id which was assigned by the service.
  """
    projectId = _messages.StringField(1, required=True)
    testMatrixId = _messages.StringField(2, required=True)