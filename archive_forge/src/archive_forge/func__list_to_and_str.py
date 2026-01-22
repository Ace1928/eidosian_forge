from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.six import integer_types, string_types
from jinja2.exceptions import TemplateSyntaxError
def _list_to_and_str(lyst):
    """Convert a list to a command delimited string
    with the last entry being an and

    :param lyst: The list to turn into a str
    :type lyst: list
    :return: The nicely formatted string
    :rtype: str
    """
    res = '{most} and {last}'.format(most=', '.join(lyst[:-1]), last=lyst[-1])
    return res