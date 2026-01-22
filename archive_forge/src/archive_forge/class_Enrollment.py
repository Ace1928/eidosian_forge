from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Enrollment(_messages.Message):
    """In case of success client will be notified with HTTP 200 response code
  but for failure scenario relevant exception message is thrown with the
  corresponding response code

  Fields:
    destinationDetails: Output only. The locations where the generated reports
      can be uploaded.
    name: Identifier. The name of this Enrollment, in the format of scope
      given in request.
  """
    destinationDetails = _messages.MessageField('DestinationDetails', 1, repeated=True)
    name = _messages.StringField(2)