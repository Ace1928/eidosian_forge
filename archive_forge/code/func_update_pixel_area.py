import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
@observe('scan_width, scan_height, image')
def update_pixel_area(self, event):
    if self.image.size > 0:
        self.pixel_area = self.scan_height * self.scan_width / self.image.size
    else:
        self.pixel_area = 0