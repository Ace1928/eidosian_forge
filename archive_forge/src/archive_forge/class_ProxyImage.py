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
class ProxyImage(Image):
    """Image returned by the Loader.image() function.

    :Properties:
        `loaded`: bool, defaults to False
            This value may be True if the image is already cached.

    :Events:
        `on_load`
            Fired when the image is loaded or changed.
        `on_error`
            Fired when the image cannot be loaded.
            `error`: Exception data that occurred
    """
    __events__ = ('on_load', 'on_error')

    def __init__(self, arg, **kwargs):
        loaded = kwargs.pop('loaded', False)
        super(ProxyImage, self).__init__(arg, **kwargs)
        self.loaded = loaded

    def on_load(self):
        pass

    def on_error(self, error):
        pass