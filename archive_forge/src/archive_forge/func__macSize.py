import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _macSize():
    return (core_graphics.CGDisplayPixelsWide(core_graphics.CGMainDisplayID()), core_graphics.CGDisplayPixelsHigh(core_graphics.CGMainDisplayID()))