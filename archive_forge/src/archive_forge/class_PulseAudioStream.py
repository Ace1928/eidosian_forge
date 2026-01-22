import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioStream(PulseAudioMainloopChild):
    """PulseAudio audio stream."""
    _state_name = {pa.PA_STREAM_UNCONNECTED: 'Unconnected', pa.PA_STREAM_CREATING: 'Creating', pa.PA_STREAM_READY: 'Ready', pa.PA_STREAM_FAILED: 'Failed', pa.PA_STREAM_TERMINATED: 'Terminated'}

    def __init__(self, context: PulseAudioContext, audio_format: 'AudioFormat') -> None:
        super().__init__(context.mainloop)
        self.state = None
        "The stream's state."
        self.index = None
        "The stream's sink index."
        self.context = weakref.ref(context)
        self._cb_write = pa.pa_stream_request_cb_t(0)
        self._cb_underflow = pa.pa_stream_notify_cb_t(0)
        self._cb_state = pa.pa_stream_notify_cb_t(self._state_callback)
        self._cb_moved = pa.pa_stream_notify_cb_t(self._moved_callback)
        self._pa_stream = pa.pa_stream_new_with_proplist(context._pa_context, f'{id(self):X}'.encode('utf-8'), self.create_sample_spec(audio_format), None, None)
        context.check_not_null(self._pa_stream)
        pa.pa_stream_set_state_callback(self._pa_stream, self._cb_state, None)
        pa.pa_stream_set_moved_callback(self._pa_stream, self._cb_moved, None)
        self._refresh_state()

    def create_sample_spec(self, audio_format: 'AudioFormat') -> pa.pa_sample_spec:
        """
        Create a PulseAudio sample spec from pyglet audio format.
        """
        _FORMATS = {('little', 8): pa.PA_SAMPLE_U8, ('big', 8): pa.PA_SAMPLE_U8, ('little', 16): pa.PA_SAMPLE_S16LE, ('big', 16): pa.PA_SAMPLE_S16BE, ('little', 24): pa.PA_SAMPLE_S24LE, ('big', 24): pa.PA_SAMPLE_S24BE}
        fmt = (sys.byteorder, audio_format.sample_size)
        if fmt not in _FORMATS:
            raise MediaException(f'Unsupported sample size/format: {fmt}')
        sample_spec = pa.pa_sample_spec()
        sample_spec.format = _FORMATS[fmt]
        sample_spec.rate = audio_format.sample_rate
        sample_spec.channels = audio_format.channels
        return sample_spec

    def delete(self) -> None:
        """If connected, disconnect, and delete the stream."""
        context = self.context()
        if context is None:
            assert _debug('No active context anymore. Cannot disconnect the stream')
            self._pa_stream = None
            return
        if self._pa_stream is None:
            assert _debug('No stream to delete.')
            return
        assert _debug('Delete PulseAudioStream')
        if not self.is_unconnected:
            assert _debug('PulseAudioStream: disconnecting')
            context.check(pa.pa_stream_disconnect(self._pa_stream))
            while not (self.is_terminated or self.is_failed):
                self.mainloop.wait()
        self._disconnect_callbacks()
        pa.pa_stream_unref(self._pa_stream)
        self._pa_stream = None

    @property
    def is_unconnected(self) -> bool:
        return self.state == pa.PA_STREAM_UNCONNECTED

    @property
    def is_creating(self) -> bool:
        return self.state == pa.PA_STREAM_CREATING

    @property
    def is_ready(self) -> bool:
        return self.state == pa.PA_STREAM_READY

    @property
    def is_failed(self) -> bool:
        return self.state == pa.PA_STREAM_FAILED

    @property
    def is_terminated(self) -> bool:
        return self.state == pa.PA_STREAM_TERMINATED

    def get_writable_size(self) -> int:
        assert self._pa_stream is not None
        r = pa.pa_stream_writable_size(self._pa_stream)
        if r == PA_INVALID_WRITABLE_SIZE:
            self.context().raise_error()
        return r

    def is_corked(self) -> bool:
        assert self._pa_stream is not None
        r = pa.pa_stream_is_corked(self._pa_stream)
        self.context().check(r)
        return bool(r)

    def get_sample_spec(self) -> pa.pa_sample_spec:
        assert self._pa_stream is not None
        return pa.pa_stream_get_sample_spec(self._pa_stream)[0]

    def connect_playback(self, tlength: int=_UINT32_MAX, minreq: int=_UINT32_MAX) -> None:
        context = self.context()
        assert self._pa_stream is not None
        assert context is not None
        device = None
        buffer_attr = pa.pa_buffer_attr()
        buffer_attr.fragsize = _UINT32_MAX
        buffer_attr.maxlength = _UINT32_MAX
        buffer_attr.tlength = tlength
        buffer_attr.prebuf = _UINT32_MAX
        buffer_attr.minreq = minreq
        flags = pa.PA_STREAM_START_CORKED | pa.PA_STREAM_INTERPOLATE_TIMING | pa.PA_STREAM_VARIABLE_RATE
        volume = None
        sync_stream = None
        context.check(pa.pa_stream_connect_playback(self._pa_stream, device, buffer_attr, flags, volume, sync_stream))
        while not self.is_ready and (not self.is_failed):
            self.mainloop.wait()
        if not self.is_ready:
            context.raise_error()
        self._refresh_sink_index()
        assert _debug('PulseAudioStream: Playback connected')

    def begin_write(self, nbytes: Optional[int]=None) -> Tuple[ctypes.c_void_p, int]:
        context = self.context()
        assert context is not None
        addr = ctypes.c_void_p()
        nbytes_st = ctypes.c_size_t(_SIZE_T_MAX if nbytes is None else nbytes)
        context.check(pa.pa_stream_begin_write(self._pa_stream, ctypes.byref(addr), ctypes.byref(nbytes_st)))
        context.check_ptr_not_null(addr)
        assert _debug(f'PulseAudioStream: begin_write nbytes={nbytes} nbytes_n={nbytes_st.value}')
        return (addr, nbytes_st.value)

    def cancel_write(self) -> None:
        self.context().check(pa.pa_stream_cancel_write(self._pa_stream))

    def write(self, data, length: int, seek_mode=pa.PA_SEEK_RELATIVE) -> int:
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        assert self.is_ready
        assert _debug(f'PulseAudioStream: writing {length} bytes')
        context.check(pa.pa_stream_write(self._pa_stream, data, length, pa.pa_free_cb_t(0), 0, seek_mode))
        return length

    def update_timing_info(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_update_timing_info(self._pa_stream, clump.pa_callback, None))

    def get_timing_info(self) -> Optional[pa.pa_timing_info]:
        """
        Retrieves the stream's timing_info struct,
        or None if it does not exist.
        Note that ctypes creates a copy of the struct, meaning it will
        be safe to use with an unlocked mainloop.
        """
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        timing_info = pa.pa_stream_get_timing_info(self._pa_stream)
        return timing_info.contents if timing_info else None

    def trigger(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_trigger(self._pa_stream, clump.pa_callback, None))

    def prebuf(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_prebuf(self._pa_stream, clump.pa_callback, None))

    def flush(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_flush(self._pa_stream, clump.pa_callback, None))

    def resume(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        return self._cork(False, callback)

    def pause(self, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        return self._cork(True, callback)

    def _cork(self, pause: Union[int, bool], callback: PulseAudioContextSuccessCallback) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_cork(self._pa_stream, pause, clump.pa_callback, None))

    def update_sample_rate(self, sample_rate: int, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
        context = self.context()
        assert context is not None
        assert self._pa_stream is not None
        clump = PulseAudioStreamSuccessCallbackLump(context, callback)
        return PulseAudioOperation(clump, pa.pa_stream_update_sample_rate(self._pa_stream, sample_rate, clump.pa_callback, None))

    def set_write_callback(self, f: PulseAudioStreamRequestCallback) -> None:
        self._cb_write = pa.pa_stream_request_cb_t(f)
        pa.pa_stream_set_write_callback(self._pa_stream, self._cb_write, None)

    def set_underflow_callback(self, f: PulseAudioStreamNotifyCallback) -> None:
        self._cb_underflow = pa.pa_stream_notify_cb_t(f)
        pa.pa_stream_set_underflow_callback(self._pa_stream, self._cb_underflow, None)

    def _connect_callbacks(self) -> None:
        s = self._pa_stream
        pa.pa_stream_set_underflow_callback(s, self._cb_underflow, None)
        pa.pa_stream_set_write_callback(s, self._cb_write, None)
        pa.pa_stream_set_state_callback(s, self._cb_state, None)
        pa.pa_stream_set_moved_callback(s, self._cb_moved, None)

    def _disconnect_callbacks(self) -> None:
        s = self._pa_stream
        pa.pa_stream_set_underflow_callback(s, pa.pa_stream_notify_cb_t(0), None)
        pa.pa_stream_set_write_callback(s, pa.pa_stream_request_cb_t(0), None)
        pa.pa_stream_set_state_callback(s, pa.pa_stream_notify_cb_t(0), None)
        pa.pa_stream_set_moved_callback(s, pa.pa_stream_notify_cb_t(0), None)

    def _state_callback(self, _stream, _userdata) -> None:
        self._refresh_state()
        assert _debug(f'PulseAudioStream: state changed to {self._state_name[self.state]}')
        self.mainloop.signal()

    def _moved_callback(self, _stream, _userdata) -> None:
        self._refresh_sink_index()
        assert _debug(f'PulseAudioStream: moved to new index {self.index}')

    def _refresh_sink_index(self) -> None:
        self.index = pa.pa_stream_get_index(self._pa_stream)
        if self.index == PA_INVALID_INDEX:
            self.context().raise_error()

    def _refresh_state(self) -> None:
        self.state = pa.pa_stream_get_state(self._pa_stream)