from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
class OpenALDriver(AbstractAudioDriver):

    def __init__(self, device_name: Optional[str]=None) -> None:
        super().__init__()
        self.device = interface.OpenALDevice(device_name)
        self.context = self.device.create_context()
        self.context.make_current()
        self._listener = OpenALListener(self)
        self.worker = PlayerWorkerThread()
        self.worker.start()

    def create_audio_player(self, source: 'Source', player: 'Player') -> 'OpenALAudioPlayer':
        assert self.device is not None, 'Device was closed'
        return OpenALAudioPlayer(self, source, player)

    def delete(self) -> None:
        if self.context is None:
            assert _debug('Duplicate OpenALDriver.delete(), ignoring')
            return
        assert _debug('Delete OpenALDriver')
        self.worker.stop()
        self.context.delete_sources()
        self.device.buffer_pool.delete()
        self.context.delete()
        self.device.close()
        self.context = None

    def have_version(self, major: int, minor: int) -> bool:
        return (major, minor) <= self.get_version()

    def get_version(self) -> Tuple[int, int]:
        assert self.device is not None, 'Device was closed'
        return self.device.get_version()

    def get_extensions(self) -> List[str]:
        assert self.device is not None, 'Device was closed'
        return self.device.get_extensions()

    def have_extension(self, extension: str) -> bool:
        return extension in self.get_extensions()

    def get_listener(self) -> 'OpenALListener':
        return self._listener