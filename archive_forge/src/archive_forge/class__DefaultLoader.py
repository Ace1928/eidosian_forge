import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
class _DefaultLoader(Loader):

    @property
    def path(self):
        return path

    @path.setter
    def path(self, value):
        global path
        path = value