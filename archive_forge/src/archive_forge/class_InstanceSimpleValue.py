import unittest
from traits.api import (
class InstanceSimpleValue(InstanceValueListener):
    ref = Instance(ArgCheckBase, ())