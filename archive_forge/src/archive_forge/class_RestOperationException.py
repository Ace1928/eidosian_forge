from __future__ import absolute_import, division, print_function
import json
import os
import re
import traceback
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import Request
class RestOperationException(Exception):
    """ Encapsulate a REST API error """

    def __init__(self, error):
        self.status = to_native(error.get('status', None))
        self.errors = [to_native(err.get('message')) for err in error.get('errors', {})]
        self.message = to_native(' '.join(self.errors))