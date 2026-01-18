import time
from testtools import content
from openstack.tests.functional import base
def wait_for_create(client, client_args, check_interval=1, timeout=60):
    """Wait for an OpenStack resource to be created

    :param client: An uncalled client resource to be called with resource_args
    :param client_args: Arguments to be passed to client
    :param name: Name of the resource (for logging)
    :param check_interval: Interval between checks
    :param timeout: Time in seconds to wait for status to update.
    :returns: True if openstack.exceptions.NotFoundException is caught
    :raises: TimeoutException

    """
    resource = client(**client_args)
    start = time.time()
    while not resource:
        time.sleep(check_interval)
        resource = client(**client_args)
        timed_out = time.time() - start >= timeout
        if not resource and timed_out:
            return False
    return True