import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union
class FakeClass:

    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.__new__ = self.fake_new

    def __repr__(self):
        return f'{self.module}.{self.name}'

    def __call__(self, *args):
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        return FakeObject(self.module, self.name, args[1:])