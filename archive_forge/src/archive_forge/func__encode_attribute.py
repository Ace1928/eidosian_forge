import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _encode_attribute(self, name, type_):
    """(INTERNAL) Encodes an attribute line.

        The attribute follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        This method must receive a the name of the attribute and its type, if
        the attribute type is nominal, ``type`` must be a list of values.

        :param name: a string.
        :param type_: a string or a list of string.
        :return: a string with the encoded attribute declaration.
        """
    for char in ' %{},':
        if char in name:
            name = '"%s"' % name
            break
    if isinstance(type_, (tuple, list)):
        type_tmp = ['%s' % encode_string(type_k) for type_k in type_]
        type_ = '{%s}' % ', '.join(type_tmp)
    return '%s %s %s' % (_TK_ATTRIBUTE, name, type_)