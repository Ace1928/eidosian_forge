from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
class DebugObject(object):
    """Base class for debug api wrappers."""
    _client_lock = threading.Lock()
    SNAPSHOT_TYPE = 'SNAPSHOT'
    LOGPOINT_TYPE = 'LOGPOINT'

    def BreakpointAction(self, type_name):
        if type_name == self.SNAPSHOT_TYPE:
            return self._debug_messages.Breakpoint.ActionValueValuesEnum.CAPTURE
        if type_name == self.LOGPOINT_TYPE:
            return self._debug_messages.Breakpoint.ActionValueValuesEnum.LOG
        raise errors.InvalidBreakpointTypeError(type_name)
    CLIENT_VERSION = 'google.com/gcloud/{0}'.format(config.CLOUD_SDK_VERSION)

    def __init__(self, debug_client=None, debug_messages=None, resource_client=None, resource_messages=None):
        """Sets up class with instantiated api client."""
        self._debug_client = debug_client or apis.GetClientInstance('clouddebugger', 'v2')
        self._debug_messages = debug_messages or apis.GetMessagesModule('clouddebugger', 'v2')
        self._resource_client = resource_client or apis.GetClientInstance('cloudresourcemanager', 'v1beta1')
        self._resource_messages = resource_messages or apis.GetMessagesModule('cloudresourcemanager', 'v1beta1')
        self._resource_parser = resources.REGISTRY.Clone()
        self._resource_parser.RegisterApiByName('clouddebugger', 'v2')