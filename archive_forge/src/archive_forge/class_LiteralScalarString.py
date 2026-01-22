from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor
class LiteralScalarString(ScalarString):
    __slots__ = 'comment'
    style = '|'

    def __new__(cls, value, anchor=None):
        return ScalarString.__new__(cls, value, anchor=anchor)