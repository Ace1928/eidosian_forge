from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.image import ImageLoader, Image
from kivy.config import Config
from kivy.utils import platform
from collections import deque
from time import sleep
from os.path import join
from os import write, close, unlink, environ
import threading
import mimetypes
def _get_error_image(self):
    if not self._error_image:
        error_png_fn = join('atlas://data/images/defaulttheme/image-missing')
        self._error_image = ImageLoader.load(filename=error_png_fn)
    return self._error_image