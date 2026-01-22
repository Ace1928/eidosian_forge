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
def _set_max_upload_per_frame(self, num):
    if num is not None and num < 1:
        raise Exception('Must have at least 1 image processing per image')
    self._max_upload_per_frame = num