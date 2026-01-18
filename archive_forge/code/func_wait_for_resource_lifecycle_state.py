from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def wait_for_resource_lifecycle_state(client, module, wait_applicable, kwargs_get, get_fn, get_param, resource, states, resource_type=None):
    """
    A utility function to  wait for the resource to get into the state as specified in
    the module options.
    :param client: OCI service client instance to call the service periodically to retrieve data.
                   e.g. VirtualNetworkClient
    :param module: Instance of AnsibleModule.
    :param wait_applicable: Specifies if wait for create is applicable for this resource
    :param kwargs_get: Dictionary containing arguments to be used to call the get function which requires multiple arguments.
    :param get_fn: Function in the SDK to get the resource. e.g. virtual_network_client.get_vcn
    :param get_param: Name of the argument in the SDK get function. e.g. "vcn_id"
    :param resource_type: Type of the resource to be created. e.g. "vcn"
    :param states: List of lifecycle states to watch for while waiting after create_fn is called.
                   e.g. [module.params['wait_until'], "FAULTY"]
    :return: A dictionary containing the resource & the "changed" status. e.g. {"vcn":{x:y}, "changed":True}
    """
    if wait_applicable and module.params.get('wait', None):
        if resource_type == 'compartment':
            _debug('Pausing execution for permission on the newly created compartment to be ready.')
            time.sleep(15)
        if kwargs_get:
            _debug('Waiting for resource to reach READY state. get_args: {0}'.format(kwargs_get))
            response_get = call_with_backoff(get_fn, **kwargs_get)
        else:
            _debug('Waiting for resource with id {0} to reach READY state.'.format(resource['id']))
            response_get = call_with_backoff(get_fn, **{get_param: resource['id']})
        if states is None:
            states = module.params.get('wait_until') or DEFAULT_READY_STATES
        resource = to_dict(oci.wait_until(client, response_get, evaluate_response=lambda r: r.data.lifecycle_state in states, max_wait_seconds=module.params.get('wait_timeout', MAX_WAIT_TIMEOUT_IN_SECONDS)).data)
    return resource