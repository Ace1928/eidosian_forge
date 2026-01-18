import copy
import ctypes
import ctypes.util
import os
import sys
from .exceptions import DecodeError
from .base import AudioFile
Close the audio file and free associated memory.