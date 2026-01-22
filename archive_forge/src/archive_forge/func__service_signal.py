from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _service_signal(self):
    """Service the signal, when necessary.

        This method must be called repeatedly by subclasses to update the
        state of the signals that require polling, which are the ones based on
        Swift temp URLs and Zaqar queues. The "NO_SIGNAL" case is also handled
        here by triggering the signal once per call.
        """
    if self._signal_transport_temp_url():
        self._service_swift_signal()
    elif self._signal_transport_zaqar():
        self._service_zaqar_signal()
    elif self._signal_transport_none():
        self.signal(details={})