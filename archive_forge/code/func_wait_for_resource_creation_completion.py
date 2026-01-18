from __future__ import (absolute_import, division, print_function)
import time
def wait_for_resource_creation_completion(oneandone_conn, resource_type, resource_id, wait_timeout, wait_interval):
    """
    Waits for the resource create operation to complete based on the timeout period.
    """
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time():
        time.sleep(wait_interval)
        resource = get_resource(oneandone_conn, resource_type, resource_id)
        if resource_type == OneAndOneResources.server:
            resource_state = resource['status']['state']
        else:
            resource_state = resource['state']
        if resource_type == OneAndOneResources.server and resource_state.lower() == 'powered_on' or (resource_type != OneAndOneResources.server and resource_state.lower() == 'active'):
            return
        elif resource_state.lower() == 'failed':
            raise Exception('%s creation failed for %s' % (resource_type, resource_id))
        elif resource_state.lower() in ('active', 'enabled', 'deploying', 'configuring'):
            continue
        else:
            raise Exception('Unknown %s state %s' % (resource_type, resource_state))
    raise Exception('Timed out waiting for %s completion for %s' % (resource_type, resource_id))