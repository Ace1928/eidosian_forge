import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class TestIntBaseRangeCompound(CommonRangeTests, unittest.TestCase):

    def setUp(self):
        self.model = ModelWithBaseRangeCompound()