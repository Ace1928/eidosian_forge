import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __WriteProtoServiceDeclaration(self, printer, name, method_info_map):
    """Write a single service declaration to a proto file."""
    printer()
    printer('service %s {', self.__GetServiceClassName(name))
    with printer.Indent():
        for method_name, method_info in method_info_map.items():
            for line in textwrap.wrap(method_info.description, printer.CalculateWidth() - 3):
                printer('// %s', line)
            printer('rpc %s (%s) returns (%s);', method_name, method_info.request_type_name, method_info.response_type_name)
    printer('}')