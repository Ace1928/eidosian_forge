from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultinjectiontestingProjectsLocationsJobsStopRequest(_messages.Message):
    """A FaultinjectiontestingProjectsLocationsJobsStopRequest object.

  Fields:
    name: Required. Name of the resource
    stopJobRequest: A StopJobRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    stopJobRequest = _messages.MessageField('StopJobRequest', 2)