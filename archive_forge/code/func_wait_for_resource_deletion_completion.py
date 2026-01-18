from __future__ import (absolute_import, division, print_function)
import time
def wait_for_resource_deletion_completion(oneandone_conn, resource_type, resource_id, wait_timeout, wait_interval):
    """
    Waits for the resource delete operation to complete based on the timeout period.
    """
    wait_timeout = time.time() + wait_timeout
    while wait_timeout > time.time():
        time.sleep(wait_interval)
        logs = oneandone_conn.list_logs(q='DELETE', period='LAST_HOUR', sort='-start_date')
        if resource_type == OneAndOneResources.server:
            _type = 'VM'
        elif resource_type == OneAndOneResources.private_network:
            _type = 'PRIVATENETWORK'
        else:
            raise Exception('Unsupported wait_for delete operation for %s resource' % resource_type)
        for log in logs:
            if log['resource']['id'] == resource_id and log['action'] == 'DELETE' and (log['type'] == _type) and (log['status']['state'] == 'OK'):
                return
    raise Exception('Timed out waiting for %s deletion for %s' % (resource_type, resource_id))