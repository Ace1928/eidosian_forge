from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
class FtdServerError(Exception):

    def __init__(self, response, code):
        super(FtdServerError, self).__init__(response)
        self.response = response
        self.code = code