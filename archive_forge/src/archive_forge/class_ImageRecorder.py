from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class ImageRecorder(Recorder):
    """Creates a recorder which allows to grab an Image from a MediaStream widget.
    """
    _model_name = Unicode('ImageRecorderModel').tag(sync=True)
    _view_name = Unicode('ImageRecorderView').tag(sync=True)
    image = Instance(Image).tag(sync=True, **widget_serialization)
    format = Unicode('png', help='The format of the image.').tag(sync=True)
    _width = Unicode().tag(sync=True)
    _height = Unicode().tag(sync=True)

    def __init__(self, format='png', filename=Recorder.filename.default_value, recording=False, autosave=False, **kwargs):
        super(ImageRecorder, self).__init__(format=format, filename=filename, recording=recording, autosave=autosave, **kwargs)
        if 'image' not in kwargs:
            self.image.observe(self._check_autosave, 'value')

    @traitlets.default('image')
    def _default_image(self):
        return Image(width=self._width, height=self._height, format=self.format)

    @observe('_width')
    def _update_image_width(self, change):
        self.image.width = self._width

    @observe('_height')
    def _update_image_height(self, change):
        self.image.height = self._height

    @observe('format')
    def _update_image_format(self, change):
        self.image.format = self.format

    @observe('image')
    def _bind_image(self, change):
        if change.old:
            change.old.unobserve(self._check_autosave, 'value')
        change.new.observe(self._check_autosave, 'value')

    def _check_autosave(self, change):
        if len(self.image.value) and self.autosave:
            self.save()

    def save(self, filename=None):
        """Save the image to a file, if no filename is given it is based on the filename trait and the format.

        >>> recorder = ImageRecorder(filename='test', format='png')
        >>> ...
        >>> recorder.save()  # will save to test.png
        >>> recorder.save('foo')  # will save to foo.png
        >>> recorder.save('foo.dat')  # will save to foo.dat

        """
        filename = filename or self.filename
        if '.' not in filename:
            filename += '.' + self.format
        if len(self.image.value) == 0:
            raise ValueError('No data, did you record anything?')
        with open(filename, 'wb') as f:
            f.write(self.image.value)