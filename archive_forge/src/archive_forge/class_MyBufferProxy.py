import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
class MyBufferProxy(BufferProxy):

    def __repr__(self):
        return f'*{BufferProxy.__repr__(self)}*'