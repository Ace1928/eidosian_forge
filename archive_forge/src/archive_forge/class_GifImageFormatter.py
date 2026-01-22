import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
class GifImageFormatter(ImageFormatter):
    """
    Create a GIF image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 1.0
    """
    name = 'img_gif'
    aliases = ['gif']
    filenames = ['*.gif']
    default_image_format = 'gif'