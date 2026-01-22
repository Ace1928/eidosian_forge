import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class ListAverageAdapter(Adapter):

    def get_average(self):
        value = self.adaptee.get_list()
        if len(value) == 0:
            return 0.0
        average = 0.0
        for item in value:
            average += item
        return average / len(value)