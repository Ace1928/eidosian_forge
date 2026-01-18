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

    Returns a resource filtered by identifier from a list of resources. This method should be
    used as an alternative of 'get resource' method when 'get resource' is nor provided by
    resource api. This method returns a wrapper of response object but that should not be
    used as an input to 'wait_until' utility as this is only a partial wrapper of response object.
    :param module The AnsibleModule representing the options provided by the user
    :param list_resource_fn The function which lists all the resources
    :param target_resource_id The identifier of the resource which should be filtered from the list
    :param kwargs A map of arguments consisting of values based on which requested resource should be searched
    :return: A custom wrapper which partially wraps a response object where the data field contains the target
     resource, if found.
    