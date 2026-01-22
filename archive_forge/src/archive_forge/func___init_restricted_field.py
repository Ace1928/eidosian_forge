import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
@classmethod
def __init_restricted_field(cls, attr_name, field):

    def getter(self):
        val = self.__data.get(field.name)
        if field.from_str is not None:
            return field.from_str(val)
        return val

    def setter(self, val):
        if val is not None and field.to_str is not None:
            val = field.to_str(val)
        if val is None:
            if field.allow_none:
                if field.name in self.__data:
                    del self.__data[field.name]
            else:
                raise TypeError('value must not be None')
        else:
            self.__data[field.name] = val
    setattr(cls, attr_name, property(getter, setter, None, field.name))