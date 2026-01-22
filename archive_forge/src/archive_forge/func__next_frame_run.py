from threading import Thread
from queue import Queue, Empty, Full
from kivy.clock import Clock, mainthread
from kivy.logger import Logger
from kivy.core.video import VideoBase
from kivy.graphics import Rectangle, BindTexture
from kivy.graphics.texture import Texture
from kivy.graphics.fbo import Fbo
from kivy.weakmethod import WeakMethod
import time
def _next_frame_run(self, ffplayer):
    sleep = time.sleep
    trigger = self._trigger
    did_dispatch_eof = False
    wait_for_wakeup = self._wait_for_wakeup
    seek_queue = self._seek_queue
    while not self._ffplayer_need_quit:
        src_pix_fmt = ffplayer.get_metadata().get('src_pix_fmt')
        if not src_pix_fmt:
            wait_for_wakeup(0.005)
            continue
        if src_pix_fmt in (b'yuv420p', 'yuv420p'):
            self._out_fmt = 'yuv420p'
            ffplayer.set_output_pix_fmt(self._out_fmt)
        break
    if self._ffplayer_need_quit:
        ffplayer.close_player()
        return
    self._ffplayer = ffplayer
    self._finish_setup()
    while not self._ffplayer_need_quit:
        seek_happened = False
        if seek_queue:
            vals = seek_queue[:]
            del seek_queue[:len(vals)]
            percent, precise = vals[-1]
            ffplayer.seek(percent * ffplayer.get_metadata()['duration'], relative=False, accurate=precise)
            seek_happened = True
            did_dispatch_eof = False
            self._next_frame = None
        if seek_happened and ffplayer.get_pause():
            ffplayer.set_volume(0.0)
            ffplayer.set_pause(False)
            try:
                to_skip = 6
                while True:
                    frame, val = ffplayer.get_frame(show=False)
                    if val in ('paused', 'eof'):
                        break
                    if seek_queue:
                        break
                    if frame is None:
                        sleep(0.005)
                        continue
                    to_skip -= 1
                    if to_skip == 0:
                        break
                frame, val = ffplayer.get_frame(force_refresh=True)
            finally:
                ffplayer.set_pause(bool(self._state == 'paused'))
                ffplayer.set_volume(self._volume)
        else:
            frame, val = ffplayer.get_frame()
        if val == 'eof':
            if not did_dispatch_eof:
                self._do_eos()
                did_dispatch_eof = True
            wait_for_wakeup(None)
        elif val == 'paused':
            did_dispatch_eof = False
            wait_for_wakeup(None)
        else:
            did_dispatch_eof = False
            if frame:
                self._next_frame = frame
                trigger()
            else:
                val = val if val else 1 / 30.0
            wait_for_wakeup(val)
    ffplayer.close_player()