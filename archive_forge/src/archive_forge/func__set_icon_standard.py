import pygame
from kivy.compat import PY2
from kivy.core.window import WindowBase
from kivy.core import CoreCriticalException
from os import environ
from os.path import exists, join
from kivy.config import Config
from kivy import kivy_data_dir
from kivy.base import ExceptionManager
from kivy.logger import Logger
from kivy.base import stopTouchApp, EventLoop
from kivy.utils import platform, deprecated
from kivy.resources import resource_find
def _set_icon_standard(self, filename):
    if PY2:
        try:
            im = pygame.image.load(filename)
        except UnicodeEncodeError:
            im = pygame.image.load(filename.encode('utf8'))
    else:
        im = pygame.image.load(filename)
    if im is None:
        raise Exception('Unable to load window icon (not found)')
    pygame.display.set_icon(im)