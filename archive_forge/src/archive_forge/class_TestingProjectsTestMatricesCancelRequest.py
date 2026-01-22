from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsTestMatricesCancelRequest(_messages.Message):
    """A TestingProjectsTestMatricesCancelRequest object.

  Fields:
    projectId: Cloud project that owns the test.
    testMatrixId: Test matrix that will be canceled.
  """
    projectId = _messages.StringField(1, required=True)
    testMatrixId = _messages.StringField(2, required=True)