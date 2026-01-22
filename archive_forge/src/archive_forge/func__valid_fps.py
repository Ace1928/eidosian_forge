from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@validate('max_fps')
def _valid_fps(self, proposal):
    if proposal['value'] is not None and proposal['value'] < 0:
        raise TraitError('max_fps attribute must be a positive integer')
    return proposal['value']