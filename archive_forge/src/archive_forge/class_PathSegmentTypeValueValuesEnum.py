from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathSegmentTypeValueValuesEnum(_messages.Enum):
    """[Output Only] The type of the AS Path, which can be one of the
    following values: - 'AS_SET': unordered set of autonomous systems that the
    route in has traversed - 'AS_SEQUENCE': ordered set of autonomous systems
    that the route has traversed - 'AS_CONFED_SEQUENCE': ordered set of Member
    Autonomous Systems in the local confederation that the route has traversed
    - 'AS_CONFED_SET': unordered set of Member Autonomous Systems in the local
    confederation that the route has traversed

    Values:
      AS_CONFED_SEQUENCE: <no description>
      AS_CONFED_SET: <no description>
      AS_SEQUENCE: <no description>
      AS_SET: <no description>
    """
    AS_CONFED_SEQUENCE = 0
    AS_CONFED_SET = 1
    AS_SEQUENCE = 2
    AS_SET = 3