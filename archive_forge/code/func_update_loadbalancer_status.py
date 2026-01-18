import os
import socket
import time
from oslo_serialization import jsonutils
import tenacity
from octavia_lib.api.drivers import data_models
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
def update_loadbalancer_status(self, status):
    """Update load balancer status.

        :param status: dictionary defining the provisioning status and
            operating status for load balancer objects, including pools,
            members, listeners, L7 policies, and L7 rules.
            iod (string): ID for the object.
            provisioning_status (string): Provisioning status for the object.
            operating_status (string): Operating status for the object.
        :type status: dict
        :raises: UpdateStatusError
        :returns: None
        """
    try:
        response = self._send(self.status_socket, status)
    except Exception as e:
        raise driver_exceptions.UpdateStatusError(fault_string=str(e))
    if response[constants.STATUS_CODE] != constants.DRVR_STATUS_CODE_OK:
        raise driver_exceptions.UpdateStatusError(fault_string=response.pop(constants.FAULT_STRING, None), status_object=response.pop(constants.STATUS_OBJECT, None), status_object_id=response.pop(constants.STATUS_OBJECT_ID, None), status_record=response.pop(constants.STATUS_RECORD, None))