from __future__ import (absolute_import, division, print_function)
class FactNamespace:

    def __init__(self, namespace_name):
        self.namespace_name = namespace_name

    def transform(self, name):
        """Take a text name, and transforms it as needed (add a namespace prefix, etc)"""
        return name

    def _underscore(self, name):
        return name.replace('-', '_')