from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class XA2SourceVoice:

    def __init__(self, voice, callback, channel_count, sample_size):
        self._voice_state = lib.XAUDIO2_VOICE_STATE()
        self._voice = voice
        self._callback = callback
        self.channel_count = channel_count
        self.sample_size = sample_size
        self.samples_played_at_last_recycle = 0
        if channel_count == 1:
            self._emitter = lib.X3DAUDIO_EMITTER()
            self._emitter.ChannelCount = channel_count
            self._emitter.CurveDistanceScaler = 1.0
            cone = lib.X3DAUDIO_CONE()
            cone.InnerVolume = 1.0
            self._emitter.pCone = pointer(cone)
            self._emitter.pVolumeCurve = None
        else:
            self._emitter = None

    def destroy(self):
        """Completely destroy the voice."""
        self._emitter = None
        if self._voice is not None:
            self._voice.DestroyVoice()
            self._voice = None
        self._callback = None

    def acquired(self, on_buffer_end_cb, sample_rate):
        """A voice has been acquired. Set the callback as well as its new sample
        rate.
        """
        self._callback.on_buffer_end = on_buffer_end_cb
        self._voice.SetSourceSampleRate(sample_rate)

    @property
    def buffers_queued(self):
        """Get the amount of buffers in the current voice. Adding flag for no samples played is 3x faster."""
        self._voice.GetState(ctypes.byref(self._voice_state), lib.XAUDIO2_VOICE_NOSAMPLESPLAYED)
        return self._voice_state.BuffersQueued

    @property
    def samples_played(self):
        """Get the amount of samples played by the voice."""
        self._voice.GetState(ctypes.byref(self._voice_state), 0)
        return self._voice_state.SamplesPlayed

    @property
    def volume(self):
        vol = c_float()
        self._voice.GetVolume(ctypes.byref(vol))
        return vol.value

    @volume.setter
    def volume(self, value):
        self._voice.SetVolume(value, 0)

    @property
    def is_emitter(self):
        return self._emitter is not None

    @property
    def position(self):
        if self.is_emitter:
            return (self._emitter.Position.x, self._emitter.Position.y, self._emitter.Position.z)
        else:
            return (0, 0, 0)

    @position.setter
    def position(self, position):
        if self.is_emitter:
            x, y, z = position
            self._emitter.Position.x = x
            self._emitter.Position.y = y
            self._emitter.Position.z = z

    @property
    def min_distance(self):
        """Curve distance scaler that is used to scale normalized distance curves to user-defined world units,
        and/or to exaggerate their effect."""
        if self.is_emitter:
            return self._emitter.CurveDistanceScaler
        else:
            return 0

    @min_distance.setter
    def min_distance(self, value):
        if self.is_emitter:
            if self._emitter.CurveDistanceScaler != value:
                self._emitter.CurveDistanceScaler = min(value, lib.FLT_MAX)

    @property
    def frequency(self):
        """The actual frequency ratio. If voice is 3d enabled, will be overwritten next apply3d cycle."""
        value = c_float()
        self._voice.GetFrequencyRatio(byref(value))
        return value.value

    @frequency.setter
    def frequency(self, value):
        if self.frequency == value:
            return
        self._voice.SetFrequencyRatio(value, 0)

    @property
    def cone_orientation(self):
        """The orientation of the sound emitter."""
        if self.is_emitter:
            return (self._emitter.OrientFront.x, self._emitter.OrientFront.y, self._emitter.OrientFront.z)
        else:
            return (0, 0, 0)

    @cone_orientation.setter
    def cone_orientation(self, value):
        if self.is_emitter:
            x, y, z = value
            self._emitter.OrientFront.x = x
            self._emitter.OrientFront.y = y
            self._emitter.OrientFront.z = z
    _ConeAngles = namedtuple('_ConeAngles', ['inside', 'outside'])

    @property
    def cone_angles(self):
        """The inside and outside angles of the sound projection cone."""
        if self.is_emitter:
            return self._ConeAngles(self._emitter.pCone.contents.InnerAngle, self._emitter.pCone.contents.OuterAngle)
        else:
            return self._ConeAngles(0, 0)

    def set_cone_angles(self, inside, outside):
        """The inside and outside angles of the sound projection cone."""
        if self.is_emitter:
            self._emitter.pCone.contents.InnerAngle = inside
            self._emitter.pCone.contents.OuterAngle = outside

    @property
    def cone_outside_volume(self):
        """The volume scaler of the sound beyond the outer cone."""
        if self.is_emitter:
            return self._emitter.pCone.contents.OuterVolume
        else:
            return 0

    @cone_outside_volume.setter
    def cone_outside_volume(self, value):
        if self.is_emitter:
            self._emitter.pCone.contents.OuterVolume = value

    @property
    def cone_inside_volume(self):
        """The volume scaler of the sound within the inner cone."""
        if self.is_emitter:
            return self._emitter.pCone.contents.InnerVolume
        else:
            return 0

    @cone_inside_volume.setter
    def cone_inside_volume(self, value):
        if self.is_emitter:
            self._emitter.pCone.contents.InnerVolume = value

    def flush(self):
        """Stop and removes all buffers already queued. OnBufferEnd is called for each."""
        self._voice.Stop(0, 0)
        self._voice.FlushSourceBuffers()

    def play(self):
        self._voice.Start(0, 0)

    def stop(self):
        self._voice.Stop(0, 0)

    def submit_buffer(self, x2_buffer):
        self._voice.SubmitSourceBuffer(ctypes.byref(x2_buffer), None)