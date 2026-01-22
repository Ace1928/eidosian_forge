from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendNodeValue(_messages.Message):
    """Evaluation result of a cel AST node NextTAG: 7

  Enums:
    CriticalNodeErrorsValueListEntryValuesEnum:
    NodeStateValueValuesEnum: Evaluation state of this node
    NonCriticalNodeErrorsValueListEntryValuesEnum:

  Fields:
    criticalNodeErrors: The errors included depend on the context. It is
      applicable when node_state is NODE_STATE_ERROR
    nodeState: Evaluation state of this node
    nonCriticalNodeErrors: The errors included depend on the context. Note:
      ACCESS_LEVEL_STATE_GRANTED/ACCESS_LEVEL_STATE_NOT_GRANTED access levels
      may have non_critical_node_errors errors underneath that don't block the
      evaluation.
    value: Evaluation result of this node, It is applicable when node_state is
      NODE_STATE_NORMAL.
  """

    class CriticalNodeErrorsValueListEntryValuesEnum(_messages.Enum):
        """CriticalNodeErrorsValueListEntryValuesEnum enum type.

    Values:
      NODE_ERROR_UNSPECIFIED: Reserved
      NODE_ERROR_INTERNAL_ERROR: Internal error If there is no matching error
        below, use it by default
      NODE_ERROR_DEVICE_NOT_FOUND: Device not found
      NODE_ERROR_DEVICE_STALE: Device is out of sync
      NODE_ERROR_DEVICE_CROSS_ORG: It is a cross-org device
      NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED: Caller doesn't have permission to
        device info
      NODE_ERROR_INVALID_SOURCE_IP: Source ip is not valid, from inIpRange
        function
      NODE_ERROR_INVALID_IP_SUBNETS: Ip subnets are not valid, from inIpRange
        function
      NODE_ERROR_INVALID_DEVICE_VERSION: Device min verion is not valid, from
        versionAtLeast function
      NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC: Expr error from a supported
        function type with invalid parameters e.g. 1 == true
      NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED: Caller doesn't have
        permission to auth session info
      NODE_ERROR_NO_BCE_LICENSE: User is not assigned a BCE license.
    """
        NODE_ERROR_UNSPECIFIED = 0
        NODE_ERROR_INTERNAL_ERROR = 1
        NODE_ERROR_DEVICE_NOT_FOUND = 2
        NODE_ERROR_DEVICE_STALE = 3
        NODE_ERROR_DEVICE_CROSS_ORG = 4
        NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED = 5
        NODE_ERROR_INVALID_SOURCE_IP = 6
        NODE_ERROR_INVALID_IP_SUBNETS = 7
        NODE_ERROR_INVALID_DEVICE_VERSION = 8
        NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC = 9
        NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED = 10
        NODE_ERROR_NO_BCE_LICENSE = 11

    class NodeStateValueValuesEnum(_messages.Enum):
        """Evaluation state of this node

    Values:
      NODE_STATE_UNSPECIFIED: Reserved
      NODE_STATE_NORMAL: The node state is normal, which means the evaluation
        of this node succeeds However, it doesn't mean the evaluated result is
        a boolean value.
      NODE_STATE_ERROR: Encounter error when evaluating the result of this
        node. Only return error if it is in the critical path of evaluation.
        For example, `( || true) && ` -> ``, ` || true` -> `true` `.foo` -> ``
        `foo()` -> `` ` + 1` -> ``
    """
        NODE_STATE_UNSPECIFIED = 0
        NODE_STATE_NORMAL = 1
        NODE_STATE_ERROR = 2

    class NonCriticalNodeErrorsValueListEntryValuesEnum(_messages.Enum):
        """NonCriticalNodeErrorsValueListEntryValuesEnum enum type.

    Values:
      NODE_ERROR_UNSPECIFIED: Reserved
      NODE_ERROR_INTERNAL_ERROR: Internal error If there is no matching error
        below, use it by default
      NODE_ERROR_DEVICE_NOT_FOUND: Device not found
      NODE_ERROR_DEVICE_STALE: Device is out of sync
      NODE_ERROR_DEVICE_CROSS_ORG: It is a cross-org device
      NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED: Caller doesn't have permission to
        device info
      NODE_ERROR_INVALID_SOURCE_IP: Source ip is not valid, from inIpRange
        function
      NODE_ERROR_INVALID_IP_SUBNETS: Ip subnets are not valid, from inIpRange
        function
      NODE_ERROR_INVALID_DEVICE_VERSION: Device min verion is not valid, from
        versionAtLeast function
      NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC: Expr error from a supported
        function type with invalid parameters e.g. 1 == true
      NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED: Caller doesn't have
        permission to auth session info
      NODE_ERROR_NO_BCE_LICENSE: User is not assigned a BCE license.
    """
        NODE_ERROR_UNSPECIFIED = 0
        NODE_ERROR_INTERNAL_ERROR = 1
        NODE_ERROR_DEVICE_NOT_FOUND = 2
        NODE_ERROR_DEVICE_STALE = 3
        NODE_ERROR_DEVICE_CROSS_ORG = 4
        NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED = 5
        NODE_ERROR_INVALID_SOURCE_IP = 6
        NODE_ERROR_INVALID_IP_SUBNETS = 7
        NODE_ERROR_INVALID_DEVICE_VERSION = 8
        NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC = 9
        NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED = 10
        NODE_ERROR_NO_BCE_LICENSE = 11
    criticalNodeErrors = _messages.EnumField('CriticalNodeErrorsValueListEntryValuesEnum', 1, repeated=True)
    nodeState = _messages.EnumField('NodeStateValueValuesEnum', 2)
    nonCriticalNodeErrors = _messages.EnumField('NonCriticalNodeErrorsValueListEntryValuesEnum', 3, repeated=True)
    value = _messages.MessageField('GoogleApiExprValue', 4)