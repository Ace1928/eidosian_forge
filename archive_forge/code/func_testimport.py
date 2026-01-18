import os
import sys
import platform as plf
from time import ctime
from configparser import ConfigParser
from io import StringIO
import kivy
from kivy.core import gl
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.core.camera import Camera
from kivy.core.image import ImageLoader
from kivy.core.text import Label
from kivy.core.video import Video
from kivy.config import Config
from kivy.input.factory import MotionEventFactory
def testimport(libname):
    try:
        lib = __import__(libname)
        report.append('%-20s exist at %s' % (libname, lib.__file__))
    except ImportError:
        report.append('%-20s is missing' % libname)