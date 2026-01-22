import math
import ctypes
from . import interface
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
class DirectSoundAudioPlayer(AbstractAudioPlayer):

    def __init__(self, driver, source, player):
        super().__init__(source, player)
        self.driver = driver
        self._playing = False
        self._cone_inner_angle = 360
        self._cone_outer_angle = 360
        self._play_cursor_ring = 0
        self._write_cursor_ring = 0
        self._write_cursor = 0
        self._play_cursor = 0
        self._eos_cursor = None
        self._possible_eos_cursor = 0
        self._has_underrun = False
        self._buffer_size = self._buffered_data_ideal_size
        self._ds_buffer = self.driver._ds_driver.create_buffer(source.audio_format, self._buffer_size)
        self._ds_buffer.current_position = 0
        assert self._ds_buffer.buffer_size == self._buffer_size

    def delete(self):
        if self.driver._ds_driver is not None:
            self.driver.worker.remove(self)
            self._ds_buffer.delete()

    def play(self):
        assert _debug('DirectSound play')
        if not self._playing:
            self._playing = True
            self._ds_buffer.play()
        self.driver.worker.add(self)
        assert _debug('return DirectSound play')

    def stop(self):
        assert _debug('DirectSound stop')
        self.driver.worker.remove(self)
        if self._playing:
            self._playing = False
            self._ds_buffer.stop()
        assert _debug('return DirectSound stop')

    def clear(self):
        assert _debug('DirectSound clear')
        super().clear()
        self._ds_buffer.current_position = 0
        self._play_cursor_ring = self._write_cursor_ring = 0
        self._play_cursor = self._write_cursor = 0
        self._eos_cursor = None
        self._possible_eos_cursor = 0
        self._has_underrun = False

    def get_play_cursor(self):
        return self._play_cursor

    def work(self):
        assert self._playing
        self._update_play_cursor()
        self.dispatch_media_events(self._play_cursor)
        if self._eos_cursor is None:
            self._maybe_fill()
            return
        if not self._has_underrun and self._play_cursor > self._eos_cursor:
            self._has_underrun = True
            assert _debug('DirectSoundAudioPlayer: Dispatching eos')
            MediaEvent('on_eos').sync_dispatch_to_player(self.player)
        if (used := self._get_used_buffer_space()) < self._buffered_data_comfortable_limit:
            self._write(None, self._buffer_size - used)

    def _maybe_fill(self):
        if (used := self._get_used_buffer_space()) < self._buffered_data_comfortable_limit:
            self._refill(self.source.audio_format.align(self._buffer_size - used))

    def _refill(self, size):
        """Refill the next `size` bytes in the buffer using the source.
        `size` must be aligned.
        """
        audio_data = self._get_and_compensate_audio_data(size, self._play_cursor)
        if audio_data is None:
            assert _debug('DirectSoundAudioPlayer: Out of audio data')
            if self._eos_cursor is None:
                self._eos_cursor = self._possible_eos_cursor
            self._write(None, size)
        else:
            assert _debug(f'DirectSoundAudioPlayer: Got {audio_data.length} bytes of audio data')
            self.append_events(self._write_cursor, audio_data.events)
            self._write(audio_data, size)

    def _update_play_cursor(self):
        play_cursor_ring = self._ds_buffer.current_position.play_cursor
        if play_cursor_ring < self._play_cursor_ring:
            self._play_cursor += self._buffer_size - self._play_cursor_ring
            self._play_cursor += play_cursor_ring
        else:
            self._play_cursor += play_cursor_ring - self._play_cursor_ring
        self._play_cursor_ring = play_cursor_ring

    def _get_used_buffer_space(self):
        return max(self._write_cursor - self._play_cursor, 0)

    def _write(self, audio_data, region_size):
        """Write data into the circular DSBuffer, starting at _write_cursor_ring.
        May supply None as audio_data to only write silence.
        If the audio data is not sufficient, will fill silence afterwards.
        If too much audio data is supplied, will write as much as fits.
        """
        if region_size == 0:
            return
        if audio_data is None:
            audio_size = 0
        else:
            audio_size = min(region_size, audio_data.length)
            self._possible_eos_cursor = self._write_cursor + audio_size
            audio_ptr = audio_data.pointer
        assert _debug(f'Writing {region_size}B ({audio_size}B data, {region_size - audio_size}B silence)')
        write_ptr = self._ds_buffer.lock(self._write_cursor_ring, region_size)
        a1_size = write_ptr.audio_length_1.value
        a2_size = write_ptr.audio_length_2.value
        assert 0 < region_size <= self._buffer_size
        assert region_size == a1_size + a2_size
        a2_silence = a2_size
        s = 128 if self.source.audio_format.sample_size == 8 else 0
        if audio_size < a1_size:
            if audio_size > 0:
                ctypes.memmove(write_ptr.audio_ptr_1, audio_ptr, audio_size)
            ctypes.memset(write_ptr.audio_ptr_1.value + audio_size, s, a1_size - audio_size)
        else:
            if a1_size > 0:
                ctypes.memmove(write_ptr.audio_ptr_1, audio_ptr, a1_size)
            if write_ptr.audio_ptr_2 and (a2_audio := (audio_size - a1_size)) > 0:
                ctypes.memmove(write_ptr.audio_ptr_2, audio_ptr + a1_size, a2_audio)
                a2_silence -= a2_audio
        if write_ptr.audio_ptr_2 and a2_silence > 0:
            ctypes.memset(write_ptr.audio_ptr_2.value + (a2_size - a2_silence), s, a2_silence)
        self._ds_buffer.unlock(write_ptr)
        self._write_cursor += region_size
        self._write_cursor_ring += region_size
        self._write_cursor_ring %= self._buffer_size

    def set_volume(self, volume):
        self._ds_buffer.volume = _gain2db(volume)

    def set_position(self, position):
        if self._ds_buffer.is3d:
            self._ds_buffer.position = _convert_coordinates(position)

    def set_min_distance(self, min_distance):
        if self._ds_buffer.is3d:
            self._ds_buffer.min_distance = min_distance

    def set_max_distance(self, max_distance):
        if self._ds_buffer.is3d:
            self._ds_buffer.max_distance = max_distance

    def set_pitch(self, pitch):
        frequency = int(pitch * self.source.audio_format.sample_rate)
        self._ds_buffer.frequency = frequency

    def set_cone_orientation(self, cone_orientation):
        if self._ds_buffer.is3d:
            self._ds_buffer.cone_orientation = _convert_coordinates(cone_orientation)

    def set_cone_inner_angle(self, cone_inner_angle):
        if self._ds_buffer.is3d:
            self._cone_inner_angle = int(cone_inner_angle)
            self._set_cone_angles()

    def set_cone_outer_angle(self, cone_outer_angle):
        if self._ds_buffer.is3d:
            self._cone_outer_angle = int(cone_outer_angle)
            self._set_cone_angles()

    def _set_cone_angles(self):
        inner = min(self._cone_inner_angle, self._cone_outer_angle)
        outer = max(self._cone_inner_angle, self._cone_outer_angle)
        self._ds_buffer.set_cone_angles(inner, outer)

    def set_cone_outer_gain(self, cone_outer_gain):
        if self._ds_buffer.is3d:
            volume = _gain2db(cone_outer_gain)
            self._ds_buffer.cone_outside_volume = volume

    def prefill_audio(self):
        self._maybe_fill()