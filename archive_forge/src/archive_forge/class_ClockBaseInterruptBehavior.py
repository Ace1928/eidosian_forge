from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
class ClockBaseInterruptBehavior(ClockBaseBehavior):
    """A kivy clock which can be interrupted during a frame to execute events.
    """
    interupt_next_only = False
    _event = None
    _async_event = None
    _get_min_timeout_func = None

    def __init__(self, interupt_next_only=False, **kwargs):
        super(ClockBaseInterruptBehavior, self).__init__(**kwargs)
        self._event = ThreadingEvent()
        self.interupt_next_only = interupt_next_only
        self._get_min_timeout_func = self.get_min_timeout

    def init_async_lib(self, lib):
        super(ClockBaseInterruptBehavior, self).init_async_lib(lib)
        if lib == 'trio':
            import trio
            self._async_event = trio.Event()
            self._async_event.set()
        elif lib == 'asyncio':
            import asyncio
            self._async_event = asyncio.Event()
            self._async_event.set()

    def usleep(self, microseconds):
        self._event.clear()
        self._event.wait(microseconds / 1000000.0)

    async def async_usleep(self, microseconds):
        self._async_event.clear()
        await self._async_wait_for(self._async_event.wait(), microseconds / 1000000.0)

    def on_schedule(self, event):
        fps = self._max_fps
        if not fps:
            return
        if not event.timeout or (not self.interupt_next_only and event.timeout <= 1 / fps - (self.time() - self._last_tick) + 4 / 5.0 * self.get_resolution()):
            self._event.set()
            if self._async_event:
                self._async_event.set()

    def idle(self):
        fps = self._max_fps
        event = self._event
        resolution = self.get_resolution()
        if fps > 0:
            done, sleeptime = self._check_ready(fps, resolution, 4 / 5.0 * resolution, event)
            if not done:
                event.wait(sleeptime)
        current = self.time()
        self._dt = current - self._last_tick
        self._last_tick = current
        event.clear()
        return current

    async def async_idle(self):
        fps = self._max_fps
        event = self._async_event
        resolution = self.get_resolution()
        if fps > 0:
            done, sleeptime = self._check_ready(fps, resolution, 4 / 5.0 * resolution, event)
            if not done:
                await self._async_wait_for(event.wait(), sleeptime)
            else:
                await self._async_lib.sleep(0)
        else:
            await self._async_lib.sleep(0)
        current = self.time()
        self._dt = current - self._last_tick
        self._last_tick = current
        event.clear()
        return current

    def _check_ready(self, fps, min_sleep, undershoot, event):
        if event.is_set():
            return (True, 0)
        t = self._get_min_timeout_func()
        if not t:
            return (True, 0)
        if not self.interupt_next_only:
            curr_t = self.time()
            sleeptime = min(1 / fps - (curr_t - self._last_tick), t - curr_t)
        else:
            sleeptime = 1 / fps - (self.time() - self._last_tick)
        return (sleeptime - undershoot <= min_sleep, sleeptime - undershoot)