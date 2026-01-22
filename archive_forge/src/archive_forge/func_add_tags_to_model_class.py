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
def add_tags_to_model_class(model, freeform_tags, defined_tags):
    """
    Add free-form and defined tags to a resource model.
    :param model:  A resource model instance that supports 'freeform_tags' and 'defined_tags' as attributes
    :param freeform_tags:  A dict representing the freeform_tags to be applied to the model
    :param defined_tags: A dict representing the defined_tags to be applied to the model
    :return: The updated model class with the tags specified by the user
    """
    try:
        if freeform_tags is not None:
            _debug('Model {0} set freeform tags to {1}'.format(model, freeform_tags))
            model.__setattr__('freeform_tags', freeform_tags)
        if defined_tags is not None:
            _debug('Model {0} set defined tags to {1}'.format(model, defined_tags))
            model.__setattr__('defined_tags', defined_tags)
    except AttributeError as ae:
        _debug("Model {0} doesn't support tags. Error {1}".format(model, ae))
    return model