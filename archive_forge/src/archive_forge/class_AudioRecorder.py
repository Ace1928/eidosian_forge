from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class AudioRecorder(Recorder):
    """Creates a recorder which allows to record the Audio of a MediaStream widget, play the
    record in the Notebook, and download it or turn it into an Audio widget.

    For help on supported values for the "codecs" attribute, see
    https://stackoverflow.com/questions/41739837/all-mime-types-supported-by-mediarecorder-in-firefox-and-chrome
    """
    _model_name = Unicode('AudioRecorderModel').tag(sync=True)
    _view_name = Unicode('AudioRecorderView').tag(sync=True)
    audio = Instance(Audio).tag(sync=True, **widget_serialization)
    codecs = Unicode('', help='Optional codecs for the recording, e.g. "opus".').tag(sync=True)

    def __init__(self, format='webm', filename=Recorder.filename.default_value, recording=False, autosave=False, **kwargs):
        super(AudioRecorder, self).__init__(format=format, filename=filename, recording=recording, autosave=autosave, **kwargs)
        if 'audio' not in kwargs:
            self.audio.observe(self._check_autosave, 'value')

    @traitlets.default('audio')
    def _default_audio(self):
        return Audio(format=self.format, controls=True)

    @observe('format')
    def _update_audio_format(self, change):
        self.audio.format = self.format

    @observe('audio')
    def _bind_audio(self, change):
        if change.old:
            change.old.unobserve(self._check_autosave, 'value')
        change.new.observe(self._check_autosave, 'value')

    def _check_autosave(self, change):
        if len(self.audio.value) and self.autosave:
            self.save()

    def save(self, filename=None):
        """Save the audio to a file, if no filename is given it is based on the filename trait and the format.

        >>> recorder = AudioRecorder(filename='test', format='mp3')
        >>> ...
        >>> recorder.save()  # will save to test.mp3
        >>> recorder.save('foo')  # will save to foo.mp3
        >>> recorder.save('foo.dat')  # will save to foo.dat

        """
        filename = filename or self.filename
        if '.' not in filename:
            filename += '.' + self.format
        if len(self.audio.value) == 0:
            raise ValueError('No data, did you record anything?')
        with open(filename, 'wb') as f:
            f.write(self.audio.value)