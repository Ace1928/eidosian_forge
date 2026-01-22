import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
class _NamedDataList(list):

    def __init__(self, name, *args):
        super(_NamedDataList, self).__init__(args)
        self.name = name

    def __str__(self):
        return str(self.name)