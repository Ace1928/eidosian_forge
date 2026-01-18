import datetime
import os
from traits.api import Array, Date, HasTraits, File, Float, Str, Tuple
def traits_init(self):
    self.scan_width, self.scan_height = self.scan_size