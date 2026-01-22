import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _winSize():
    return (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))