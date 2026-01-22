import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def WriteInit(self, out):
    """Write a simple __init__.py for the generated client."""
    printer = self._GetPrinter(out)
    if self.__init_wildcards_file:
        printer('"""Common imports for generated %s client library."""', self.__client_info.package)
        printer('# pylint:disable=wildcard-import')
    else:
        printer('"""Package marker file."""')
    printer()
    printer('from __future__ import absolute_import')
    printer()
    printer('import pkgutil')
    printer()
    if self.__init_wildcards_file:
        printer('from %s import *', self.__base_files_package)
        if self.__root_package == '.':
            import_prefix = '.'
        else:
            import_prefix = '%s.' % self.__root_package
        printer('from %s%s import *', import_prefix, self.__client_info.client_rule_name)
        printer('from %s%s import *', import_prefix, self.__client_info.messages_rule_name)
        printer()
    printer('__path__ = pkgutil.extend_path(__path__, __name__)')