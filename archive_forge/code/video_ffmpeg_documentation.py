from kivy.core.video import VideoBase
from kivy.graphics.texture import Texture

FFmpeg video abstraction
========================

.. versionadded:: 1.0.8

This abstraction requires ffmpeg python extensions. We have made a special
extension that is used for the android platform but can also be used on x86
platforms. The project is available at::

    http://github.com/tito/ffmpeg-android

The extension is designed for implementing a video player.
Refer to the documentation of the ffmpeg-android project for more information
about the requirements.
