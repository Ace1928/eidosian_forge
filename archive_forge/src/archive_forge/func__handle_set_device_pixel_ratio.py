import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
def _handle_set_device_pixel_ratio(self, device_pixel_ratio):
    if self._set_device_pixel_ratio(device_pixel_ratio):
        self._force_full = True
        self.draw_idle()