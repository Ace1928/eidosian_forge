import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
@provides(IList, IAverage)
class SampleAverage(HasTraits):
    data = List(Int, [100, 200, 300])

    def get_list(self):
        return self.data

    def get_average(self):
        value = self.get_list()
        if len(value) == 0:
            return 0.0
        average = 0.0
        for item in value:
            average += item
        return average / len(value)