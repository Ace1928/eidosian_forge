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
def _pygame_update_modifiers(self, mods=None):
    if mods is None:
        mods = pygame.key.get_mods()
    self._modifiers = []
    if mods & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT):
        self._modifiers.append('shift')
    if mods & (pygame.KMOD_ALT | pygame.KMOD_LALT):
        self._modifiers.append('alt')
    if mods & (pygame.KMOD_CTRL | pygame.KMOD_LCTRL):
        self._modifiers.append('ctrl')
    if mods & (pygame.KMOD_META | pygame.KMOD_LMETA):
        self._modifiers.append('meta')