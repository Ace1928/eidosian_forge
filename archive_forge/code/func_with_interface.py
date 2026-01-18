from typing import Union
from warnings import warn
from .low_level import *
def with_interface(self, interface):
    return type(self)(self.object_path, self.bus_name, interface)