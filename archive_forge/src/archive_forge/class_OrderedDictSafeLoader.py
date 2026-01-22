from __future__ import absolute_import, unicode_literals
from collections import OrderedDict
import six
import yaml
from pybtex.database import Entry, Person
from pybtex.database.input import BaseParser
class OrderedDictSafeLoader(yaml.SafeLoader):
    """
    SafeLoader that loads mappings as OrderedDicts.
    """

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping