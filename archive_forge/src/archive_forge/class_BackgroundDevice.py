import time
from multiprocessing import Process
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple
import zmq
from zmq import ENOTSOCK, ETERM, PUSH, QUEUE, Context, ZMQBindError, ZMQError, device
class BackgroundDevice(Device):
    """Base class for launching Devices in background processes and threads."""
    launcher: Any = None
    _launch_class: Any = None

    def start(self) -> None:
        self.launcher = self._launch_class(target=self.run)
        self.launcher.daemon = self.daemon
        return self.launcher.start()

    def join(self, timeout: Optional[float]=None) -> None:
        return self.launcher.join(timeout=timeout)