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

            # unhandled event !
            else:
                Logger.debug('WinPygame: Unhandled event %s' % str(event))
            