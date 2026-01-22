import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
class _NamedDataDict(dict):

    def __init__(self, **kwargs):
        if 'name' not in kwargs.keys():
            raise KeyError("@named_data expects a dictionary with a 'name' key.")
        self.name = kwargs.pop('name')
        super(_NamedDataDict, self).__init__(kwargs)

    def __str__(self):
        return str(self.name)