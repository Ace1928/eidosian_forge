import unittest
from traits.api import (
class UnionAllowStr(Union):

    def validate(self, obj, name, value):
        if isinstance(value, str):
            return value
        return super().validate(obj, name, value)