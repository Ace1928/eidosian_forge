import unittest
from traits.api import (
class InstanceDictValue(InstanceValueListener):
    ref = Instance(ArgCheckDict, ())