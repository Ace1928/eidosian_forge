import os
import socket
import time
from oslo_serialization import jsonutils
import tenacity
from octavia_lib.api.drivers import data_models
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
@tenacity.retry(stop=tenacity.stop_after_attempt(30), reraise=True, wait=tenacity.wait_exponential(multiplier=1, min=1, max=5), retry=tenacity.retry_if_exception_type(driver_exceptions.DriverAgentNotFound))
def _check_for_socket_ready(self, socket):
    if not os.path.exists(socket):
        raise driver_exceptions.DriverAgentNotFound(fault_string='Unable to open the driver agent socket: {}'.format(socket))