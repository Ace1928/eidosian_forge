from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
class _LvString(yaml.scalarstring.ScalarString):
    """Location/value string type."""
    __slots__ = ('lc', 'value')
    style = ''

    def __new__(cls, value):
        return yaml.scalarstring.ScalarString.__new__(cls, value)