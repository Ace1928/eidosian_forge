import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _XCPrintableValue(self, tabs, value, flatten_list=False):
    """Returns a representation of value that may be printed in a project file,
    mimicking Xcode's behavior.

    _XCPrintableValue can handle str and int values, XCObjects (which are
    made printable by returning their id property), and list and dict objects
    composed of any of the above types.  When printing a list or dict, and
    _should_print_single_line is False, the tabs parameter is used to determine
    how much to indent the lines corresponding to the items in the list or
    dict.

    If flatten_list is True, single-element lists will be transformed into
    strings.
    """
    printable = ''
    comment = None
    if self._should_print_single_line:
        sep = ' '
        element_tabs = ''
        end_tabs = ''
    else:
        sep = '\n'
        element_tabs = '\t' * (tabs + 1)
        end_tabs = '\t' * tabs
    if isinstance(value, XCObject):
        printable += value.id
        comment = value.Comment()
    elif isinstance(value, str):
        printable += self._EncodeString(value)
    elif isinstance(value, str):
        printable += self._EncodeString(value.encode('utf-8'))
    elif isinstance(value, int):
        printable += str(value)
    elif isinstance(value, list):
        if flatten_list and len(value) <= 1:
            if len(value) == 0:
                printable += self._EncodeString('')
            else:
                printable += self._EncodeString(value[0])
        else:
            printable = '(' + sep
            for item in value:
                printable += element_tabs + self._XCPrintableValue(tabs + 1, item, flatten_list) + ',' + sep
            printable += end_tabs + ')'
    elif isinstance(value, dict):
        printable = '{' + sep
        for item_key, item_value in sorted(value.items()):
            printable += element_tabs + self._XCPrintableValue(tabs + 1, item_key, flatten_list) + ' = ' + self._XCPrintableValue(tabs + 1, item_value, flatten_list) + ';' + sep
        printable += end_tabs + '}'
    else:
        raise TypeError("Can't make " + value.__class__.__name__ + ' printable')
    if comment:
        printable += ' ' + self._EncodeComment(comment)
    return printable