from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
Read file content.

    Try read content of file at specified path.
    :type path:  str
    :param path: Full path to location of file which should be read'ed.
    :rtype:  content
    :return: File content or 'None'
    