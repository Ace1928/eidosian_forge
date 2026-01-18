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
def update_and_wait(resource_type, client, update_fn, kwargs_update, get_fn, get_param, module, states=None, wait_applicable=True, kwargs_get=None):
    """
    A utility function to update a resource and wait for the resource to get into the state as specified in the module
    options. It wraps the create_and_wait method as apart from the method and arguments, everything else is similar.
    :param wait_applicable: Specifies if wait for create is applicable for this resource
    :param resource_type: Type of the resource to be created. e.g. "vcn"
    :param client: OCI service client instance to call the service periodically to retrieve data.
                   e.g. VirtualNetworkClient()
    :param update_fn: Function in the SDK to update the resource. e.g. virtual_network_client.update_vcn
    :param kwargs_update: Dictionary containing arguments to be used to call the update function update_fn.
    :param get_fn: Function in the SDK to get the resource. e.g. virtual_network_client.get_vcn
    :param get_param: Name of the argument in the SDK get function. e.g. "vcn_id"
    :param module: Instance of AnsibleModule.
    :param kwargs_get: Dictionary containing arguments to be used to call the get function which requires multiple arguments.
    :param states: List of lifecycle states to watch for while waiting after update_fn is called.
                   e.g. [module.params['wait_until'], "FAULTY"]
    :return: A dictionary containing the resource & the "changed" status. e.g. {"vcn":{x:y}, "changed":True}
    """
    try:
        return create_or_update_resource_and_wait(resource_type, update_fn, kwargs_update, module, wait_applicable, get_fn, get_param, states, client, kwargs_get=kwargs_get)
    except MaximumWaitTimeExceeded as ex:
        module.fail_json(msg=str(ex))
    except ServiceError as ex:
        module.fail_json(msg=ex.message)