from __future__ import absolute_import
import os
import logging
from urllib.request import urlopen
from traitlets import (
from ipywidgets import DOMWidget, Image, Video, Audio, register, widget_serialization
import ipywebrtc._version
import traitlets
@register
class MediaStream(DOMWidget):
    """Represents a media source.

    See https://developer.mozilla.org/nl/docs/Web/API/MediaStream for details
    In practice this can a stream coming from an HTMLVideoElement,
    HTMLCanvasElement (could be a WebGL canvas) or a camera/webcam/microphone
    using getUserMedia.

    The currently supported MediaStream (subclasses) are:
       * :class:`VideoStream`: A video file/data as media stream.
       * :class:`CameraStream`: Webcam/camera as media stream.
       * :class:`ImageStream`: An image as a static stream.
       * :class:`WidgetStream`: Arbitrary DOMWidget as stream.

    A MediaStream can be used with:
       * :class:`VideoRecorder`: To record a movie
       * :class:`ImageRecorder`: To create images/snapshots.
       * :class:`AudioRecorder`: To record audio.
       * :class:`WebRTCRoom` (or rather :class:`WebRTCRoomMqtt`): To stream a media stream to a (set of) peers.
    """
    _model_module = Unicode('jupyter-webrtc').tag(sync=True)
    _view_module = Unicode('jupyter-webrtc').tag(sync=True)
    _view_name = Unicode('MediaStreamView').tag(sync=True)
    _model_name = Unicode('MediaStreamModel').tag(sync=True)
    _view_module_version = Unicode(semver_range_frontend).tag(sync=True)
    _model_module_version = Unicode(semver_range_frontend).tag(sync=True)