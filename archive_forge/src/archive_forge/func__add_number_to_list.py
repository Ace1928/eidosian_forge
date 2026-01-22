import os
import shutil
import tempfile
import threading
import unittest
from traits.api import HasTraits, on_trait_change, Bool, Float, List
from traits import trait_notifiers
from traits.util.event_tracer import (
@on_trait_change('number')
def _add_number_to_list(self, value):
    self.list_of_numbers.append(value)