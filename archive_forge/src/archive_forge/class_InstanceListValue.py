import unittest
from traits.api import (
class InstanceListValue(InstanceValueListener):
    ref = Instance(ArgCheckList, ())