import time
from testtools import content
from openstack.tests.functional import base
Wait for an OpenStack resource to 404/delete

    :param client: An uncalled client resource to be called with resource_args
    :param client_args: Arguments to be passed to client
    :param name: Name of the resource (for logging)
    :param check_interval: Interval between checks
    :param timeout: Time in seconds to wait for status to update.
    :returns: True if openstack.exceptions.NotFoundException is caught
    :raises: TimeoutException

    