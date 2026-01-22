from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class XpnProjectStatusValueValuesEnum(_messages.Enum):
    """[Output Only] The role this project has in a shared VPC configuration.
    Currently, only projects with the host role, which is specified by the
    value HOST, are differentiated.

    Values:
      HOST: <no description>
      UNSPECIFIED_XPN_PROJECT_STATUS: <no description>
    """
    HOST = 0
    UNSPECIFIED_XPN_PROJECT_STATUS = 1