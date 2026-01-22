from kombu.utils.limits import TokenBucket
from celery import platforms
from celery.app import app_or_default
from celery.utils.dispatch import Signal
from celery.utils.imports import instantiate
from celery.utils.log import get_logger
from celery.utils.time import rate
from celery.utils.timer2 import Timer
class Polaroid:
    """Record event snapshots."""
    timer = None
    shutter_signal = Signal(name='shutter_signal', providing_args={'state'})
    cleanup_signal = Signal(name='cleanup_signal')
    clear_after = False
    _tref = None
    _ctref = None

    def __init__(self, state, freq=1.0, maxrate=None, cleanup_freq=3600.0, timer=None, app=None):
        self.app = app_or_default(app)
        self.state = state
        self.freq = freq
        self.cleanup_freq = cleanup_freq
        self.timer = timer or self.timer or Timer()
        self.logger = logger
        self.maxrate = maxrate and TokenBucket(rate(maxrate))

    def install(self):
        self._tref = self.timer.call_repeatedly(self.freq, self.capture)
        self._ctref = self.timer.call_repeatedly(self.cleanup_freq, self.cleanup)

    def on_shutter(self, state):
        pass

    def on_cleanup(self):
        pass

    def cleanup(self):
        logger.debug('Cleanup: Running...')
        self.cleanup_signal.send(sender=self.state)
        self.on_cleanup()

    def shutter(self):
        if self.maxrate is None or self.maxrate.can_consume():
            logger.debug('Shutter: %s', self.state)
            self.shutter_signal.send(sender=self.state)
            self.on_shutter(self.state)

    def capture(self):
        self.state.freeze_while(self.shutter, clear_after=self.clear_after)

    def cancel(self):
        if self._tref:
            self._tref()
            self._tref.cancel()
        if self._ctref:
            self._ctref.cancel()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc_info):
        self.cancel()