from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class XAudio2Driver:
    allow_3d = True
    processor = lib.XAUDIO2_DEFAULT_PROCESSOR
    category = lib.AudioCategory_GameEffects
    restart_on_error = True
    max_frequency_ratio = 2.0

    def __init__(self):
        """Creates an XAudio2 master voice and sets up 3D audio if specified. This attaches to the default audio
        device and will create a virtual audio endpoint that changes with the system. It will not recover if a
        critical error is encountered such as no more audio devices are present.
        """
        assert _debug('Constructing XAudio2Driver')
        self._listener = None
        self._xaudio2 = None
        self._dead = False
        self.lock = threading.Lock()
        self._engine_callback = XA2EngineCallback(self.lock)
        self._emitting_voices = []
        self._voice_pool = defaultdict(list)
        self._in_use = {}
        self._resetting_voices = {}
        self._players = []
        self._create_xa2()
        if self.restart_on_error:
            audio_devices = get_audio_device_manager()
            if audio_devices:
                assert _debug('Audio device instance found.')
                audio_devices.push_handlers(self)
                if audio_devices.get_default_output() is None:
                    raise ImportError('No default audio device found, can not create driver.')
                pyglet.clock.schedule_interval_soft(self._check_state, 0.5)

    def _check_state(self, dt):
        """Hack/workaround, you cannot shutdown/create XA2 within a COM callback, set a schedule to check state."""
        if self._dead is True:
            if self._xaudio2:
                self._shutdown_xaudio2()
        elif not self._xaudio2:
            self._create_xa2()
            for player in self._players:
                player.dispatch_event('on_driver_reset')
            self._players.clear()

    def on_default_changed(self, device, flow: DeviceFlow):
        if flow == DeviceFlow.OUTPUT:
            'Callback derived from the Audio Devices to help us determine when the system no longer has output.'
            if device is None:
                assert _debug('Error: Default audio device was removed or went missing.')
                self._dead = True
            elif self._dead:
                assert _debug('Warning: Default audio device added after going missing.')
                self._dead = False

    def _create_xa2(self, device_id=None):
        self._xaudio2 = lib.IXAudio2()
        try:
            lib.XAudio2Create(ctypes.byref(self._xaudio2), 0, self.processor)
        except OSError:
            raise ImportError('XAudio2 driver could not be initialized.')
        if _debug:
            debug = lib.XAUDIO2_DEBUG_CONFIGURATION()
            debug.LogThreadID = True
            debug.TraceMask = lib.XAUDIO2_LOG_ERRORS | lib.XAUDIO2_LOG_WARNINGS
            debug.BreakMask = lib.XAUDIO2_LOG_WARNINGS
            self._xaudio2.SetDebugConfiguration(ctypes.byref(debug), None)
        self._xaudio2.RegisterForCallbacks(self._engine_callback)
        self._mvoice_details = lib.XAUDIO2_VOICE_DETAILS()
        self._master_voice = lib.IXAudio2MasteringVoice()
        self._xaudio2.CreateMasteringVoice(byref(self._master_voice), lib.XAUDIO2_DEFAULT_CHANNELS, lib.XAUDIO2_DEFAULT_SAMPLERATE, 0, device_id, None, self.category)
        self._master_voice.GetVoiceDetails(byref(self._mvoice_details))
        self._x3d_handle = None
        self._dsp_settings = None
        if self.allow_3d:
            self.enable_3d()

    @property
    def active_voices(self):
        return self._in_use.keys()

    def _destroy_voices(self):
        """Destroy and clear all voice pools."""
        for list_ in self._voice_pool.values():
            for voice in list_:
                voice.destroy()
            list_.clear()
        for voice, resetter in self._resetting_voices.items():
            voice.destroy()
            resetter.destroy()
        self._resetting_voices.clear()
        for voice in self.active_voices:
            voice.destroy()
        self._in_use.clear()

    def set_device(self, device):
        """Attach XA2 with a specific device rather than the virtual device."""
        self._shutdown_xaudio2()
        self._create_xa2(device.id)
        for player in self._players:
            player.dispatch_event('on_driver_reset')
        self._players.clear()

    def _shutdown_xaudio2(self):
        """Stops and destroys all active voices, then destroys XA2 instance."""
        for player in self._in_use.values():
            player.on_driver_destroy()
            self._players.append(player.player)
        self._delete_driver()

    def _delete_driver(self):
        if self._xaudio2:
            assert _debug('XAudio2Driver: Deleting')
            if self.allow_3d:
                pyglet.clock.unschedule(self._calculate_3d_sources)
            self._destroy_voices()
            self._xaudio2.UnregisterForCallbacks(self._engine_callback)
            self._xaudio2.StopEngine()
            self._xaudio2.Release()
            self._xaudio2 = None

    def enable_3d(self):
        """Initializes the prerequisites for 3D positional audio and initializes with default DSP settings."""
        channel_mask = DWORD()
        self._master_voice.GetChannelMask(byref(channel_mask))
        self._x3d_handle = lib.X3DAUDIO_HANDLE()
        lib.X3DAudioInitialize(channel_mask.value, lib.X3DAUDIO_SPEED_OF_SOUND, self._x3d_handle)
        matrix = (FLOAT * self._mvoice_details.InputChannels)()
        self._dsp_settings = lib.X3DAUDIO_DSP_SETTINGS()
        self._dsp_settings.SrcChannelCount = 1
        self._dsp_settings.DstChannelCount = self._mvoice_details.InputChannels
        self._dsp_settings.pMatrixCoefficients = matrix
        pyglet.clock.schedule_interval_soft(self._calculate_3d_sources, 1 / 15.0)

    @property
    def volume(self):
        vol = c_float()
        self._master_voice.GetVolume(ctypes.byref(vol))
        return vol.value

    @volume.setter
    def volume(self, value):
        """Sets global volume of the master voice."""
        self._master_voice.SetVolume(value, 0)

    def _calculate_3d_sources(self, dt):
        """We calculate the 3d emitters and sources every 15 fps, committing everything after deferring all changes."""
        for source_voice in self._emitting_voices:
            self._apply3d(source_voice, 1)
        self._xaudio2.CommitChanges(1)

    def apply3d(self, source_voice):
        """Apply and immediately commit positional audio effects for the given voice."""
        if self._x3d_handle is not None:
            self._apply3d(source_voice, 2)
            self._xaudio2.CommitChanges(2)

    def _apply3d(self, source_voice, commit):
        """Calculates and sets output matrix and frequency ratio on the voice based on the listener and the voice's
           emitter. Commit determines the operation set, whether the settings are applied immediately (0) or to
           be committed together at a later time.
        """
        lib.X3DAudioCalculate(self._x3d_handle, self._listener.listener, source_voice._emitter, lib.default_dsp_calculation, self._dsp_settings)
        source_voice._voice.SetOutputMatrix(self._master_voice, 1, self._mvoice_details.InputChannels, self._dsp_settings.pMatrixCoefficients, commit)
        source_voice._voice.SetFrequencyRatio(self._dsp_settings.DopplerFactor, commit)

    def delete(self):
        self._delete_driver()
        pyglet.clock.unschedule(self._check_state)

    def get_performance(self):
        """Retrieve some basic XAudio2 performance data such as memory usage and source counts."""
        pf = lib.XAUDIO2_PERFORMANCE_DATA()
        self._xaudio2.GetPerformanceData(ctypes.byref(pf))
        return pf

    def create_listener(self):
        assert self._listener is None, 'You can only create one listener.'
        self._listener = XAudio2Listener(self)
        return self._listener

    def return_voice(self, voice, remaining_data):
        """Reset a voice and eventually return it to the pool. The voice must be stopped.
        `remaining_data` should contain the data this voice's remaining
        buffers point to.
        It will be `.clear()`ed shortly after as soon as the flush initiated
        by the driver completes in order to not have theoretical dangling
        pointers.
        """
        if voice.is_emitter:
            self._emitting_voices.remove(voice)
        self._in_use.pop(voice)
        assert _debug(f'XA2AudioDriver: Resetting {voice}...')
        voice_key = (voice.channel_count, voice.sample_size)
        resetter = _VoiceResetter(self, voice, voice_key, remaining_data)
        self._resetting_voices[voice] = resetter
        resetter.run()

    def _return_reset_voice(self, voice, voice_key):
        self._resetting_voices.pop(voice).destroy()
        self._voice_pool[voice_key].append(voice)
        assert _debug(f'XA2AudioDriver: {voice} back in pool')

    def get_source_voice(self, audio_format, player):
        """Get a source voice from the pool. Source voice creation can be slow to create/destroy.
        So pooling is recommended. We pool based on audio channels.
        A source voice handles all of the audio playing and state for a single source."""
        voice_key = (audio_format.channels, audio_format.sample_size)
        if not self._voice_pool[voice_key]:
            voice = self._create_new_voice(audio_format)
            self._voice_pool[voice_key].append(self._create_new_voice(audio_format))
        else:
            voice = self._voice_pool[voice_key].pop()
        assert voice.buffers_queued == 0
        voice.acquired(player.on_buffer_end, audio_format.sample_rate)
        if voice.is_emitter:
            self._emitting_voices.append(voice)
        self._in_use[voice] = player
        return voice

    def _create_new_voice(self, audio_format):
        """Has the driver create a new source voice for the given audio format."""
        voice = lib.IXAudio2SourceVoice()
        wfx_format = create_xa2_waveformat(audio_format)
        callback = XAudio2VoiceCallback()
        self._xaudio2.CreateSourceVoice(ctypes.byref(voice), ctypes.byref(wfx_format), 0, self.max_frequency_ratio, callback, None, None)
        return XA2SourceVoice(voice, callback, audio_format.channels, audio_format.sample_size)