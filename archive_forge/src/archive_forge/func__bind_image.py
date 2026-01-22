from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@observe('image')
def _bind_image(self, change):
    if change.old:
        change.old.unobserve(self._check_autosave, 'value')
    change.new.observe(self._check_autosave, 'value')