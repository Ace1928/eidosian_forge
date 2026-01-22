from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstallationValueValuesEnum(_messages.Enum):
    """Configures the manner in which the template library is installed on
    the cluster.

    Values:
      INSTALLATION_UNSPECIFIED: No installation strategy has been specified.
      NOT_INSTALLED: Do not install the template library.
      ALL: Install the entire template library.
    """
    INSTALLATION_UNSPECIFIED = 0
    NOT_INSTALLED = 1
    ALL = 2