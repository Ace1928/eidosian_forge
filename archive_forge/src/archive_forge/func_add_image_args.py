import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_image_args(self, image_mode):
    """
        Add additional arguments for handling images.
        """
    try:
        parlai = self.add_argument_group('ParlAI Image Preprocessing Arguments')
        parlai.add_argument('--image-size', type=int, default=256, help='resizing dimension for images', hidden=True)
        parlai.add_argument('--image-cropsize', type=int, default=224, help='crop dimension for images', hidden=True)
    except argparse.ArgumentError:
        pass