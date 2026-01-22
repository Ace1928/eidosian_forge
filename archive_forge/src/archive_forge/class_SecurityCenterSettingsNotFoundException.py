from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
class SecurityCenterSettingsNotFoundException(core_exceptions.Error):
    """Not Found Exception raised from SCC settings backend api."""
    pass