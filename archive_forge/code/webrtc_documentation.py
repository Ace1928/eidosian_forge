from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
Save the audio to a file, if no filename is given it is based on the filename trait and the format.

        >>> recorder = AudioRecorder(filename='test', format='mp3')
        >>> ...
        >>> recorder.save()  # will save to test.mp3
        >>> recorder.save('foo')  # will save to foo.mp3
        >>> recorder.save('foo.dat')  # will save to foo.dat

        