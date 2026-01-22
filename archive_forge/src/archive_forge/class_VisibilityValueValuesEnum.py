from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class VisibilityValueValuesEnum(_messages.Enum):
    """The zone's visibility: public zones are exposed to the Internet, while
    private zones are visible only to Virtual Private Cloud resources.

    Values:
      public: <no description>
      private: <no description>
    """
    public = 0
    private = 1