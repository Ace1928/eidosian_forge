from kivy.logger import Logger
from kivy.core.window import WindowBase
from kivy.base import EventLoop, ExceptionManager, stopTouchApp
from kivy.lib.vidcore_lite import bcm, egl
from os import environ

EGL Rpi Window: EGL Window provider, specialized for the Pi

Inspired by: rpi_vid_core + JF002 rpi kivy  repo
