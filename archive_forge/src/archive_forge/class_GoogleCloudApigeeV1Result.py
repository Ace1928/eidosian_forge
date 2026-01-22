from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Result(_messages.Message):
    """Result is short for "action result", could be different types identified
  by "action_result" field. Supported types: 1. DebugInfo : generic debug info
  collected by runtime recorded as a list of properties. For example, the
  contents could be virtual host info, state change result, or execution
  metadata. Required fields : properties, timestamp 2. RequestMessage:
  information of a http request. Contains headers, request URI and http
  methods type.Required fields : headers, uri, verb 3. ResponseMessage:
  information of a http response. Contains headers, reason phrase and http
  status code. Required fields : headers, reasonPhrase, statusCode 4.
  ErrorMessage: information of a http error message. Contains detail error
  message, reason phrase and status code. Required fields : content, headers,
  reasonPhrase, statusCode 5. VariableAccess: a list of variable access
  actions, can be Get, Set and Remove. Required fields : accessList

  Fields:
    ActionResult: Type of the action result. Can be one of the five:
      DebugInfo, RequestMessage, ResponseMessage, ErrorMessage, VariableAccess
    accessList: A list of variable access actions agaist the api proxy.
      Supported values: Get, Set, Remove.
    content: Error message content. for example, "content" :
      "{\\"fault\\":{\\"faultstring\\":\\"API timed
      out\\",\\"detail\\":{\\"errorcode\\":\\"flow.APITimedOut\\"}}}"
    headers: A list of HTTP headers. for example, '"headers" : [ { "name" :
      "Content-Length", "value" : "83" }, { "name" : "Content-Type", "value" :
      "application/json" } ]'
    properties: Name value pairs used for DebugInfo ActionResult.
    reasonPhrase: HTTP response phrase
    statusCode: HTTP response code
    timestamp: Timestamp of when the result is recorded. Its format is dd-mm-
      yy hh:mm:ss:xxx. For example, `"timestamp" : "12-08-19 00:31:59:960"`
    uRI: The relative path of the api proxy. for example, `"uRI" :
      "/iloveapis"`
    verb: HTTP method verb
  """
    ActionResult = _messages.StringField(1)
    accessList = _messages.MessageField('GoogleCloudApigeeV1Access', 2, repeated=True)
    content = _messages.StringField(3)
    headers = _messages.MessageField('GoogleCloudApigeeV1Property', 4, repeated=True)
    properties = _messages.MessageField('GoogleCloudApigeeV1Properties', 5)
    reasonPhrase = _messages.StringField(6)
    statusCode = _messages.StringField(7)
    timestamp = _messages.StringField(8)
    uRI = _messages.StringField(9)
    verb = _messages.StringField(10)