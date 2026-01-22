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
def _get_attributes_to_consider(exclude_attributes, model, module):
    """
    Determine the attributes to detect if an existing resource already matches the requested resource state
    :param exclude_attributes: Attributes to not consider for matching
    :param model: The model class used to create the Resource
    :param module: An instance of AnsibleModule that contains user's desires around a resource's state
    :return: A list of attributes that needs to be matched
    """
    if 'key_by' in module.params and module.params['key_by'] is not None:
        attributes_to_consider = module.params['key_by']
    else:
        attributes_to_consider = list(model.attribute_map)
        if 'freeform_tags' in attributes_to_consider:
            attributes_to_consider.remove('freeform_tags')
        if 'node_count' in attributes_to_consider:
            attributes_to_consider.remove('node_count')
    _debug('attributes to consider: {0}'.format(attributes_to_consider))
    return attributes_to_consider