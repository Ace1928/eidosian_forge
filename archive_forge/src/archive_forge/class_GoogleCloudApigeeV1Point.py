from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Point(_messages.Message):
    """Point is a group of information collected by runtime plane at critical
  points of the message flow of the processed API request. This is a list of
  supported point IDs, categorized to three major buckets. For each category,
  debug points that we are currently supporting are listed below: - Flow
  status debug points: StateChange FlowInfo Condition Execution DebugMask
  Error - Flow control debug points: FlowCallout Paused Resumed FlowReturn
  BreakFlow Error - Runtime debug points: ScriptExecutor
  FlowCalloutStepDefinition CustomTarget StepDefinition Oauth2ServicePoint
  RaiseFault NodeJS The detail information of the given debug point is stored
  in a list of results.

  Fields:
    id: Name of a step in the transaction.
    results: List of results extracted from a given debug point.
  """
    id = _messages.StringField(1)
    results = _messages.MessageField('GoogleCloudApigeeV1Result', 2, repeated=True)