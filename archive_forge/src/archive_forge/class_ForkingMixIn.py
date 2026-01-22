import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
class ForkingMixIn:
    """Mix-in class to handle each request in a new process."""
    timeout = 300
    active_children = None
    max_children = 40
    block_on_close = True

    def collect_children(self, *, blocking=False):
        """Internal routine to wait for children that have exited."""
        if self.active_children is None:
            return
        while len(self.active_children) >= self.max_children:
            try:
                pid, _ = os.waitpid(-1, 0)
                self.active_children.discard(pid)
            except ChildProcessError:
                self.active_children.clear()
            except OSError:
                break
        for pid in self.active_children.copy():
            try:
                flags = 0 if blocking else os.WNOHANG
                pid, _ = os.waitpid(pid, flags)
                self.active_children.discard(pid)
            except ChildProcessError:
                self.active_children.discard(pid)
            except OSError:
                pass

    def handle_timeout(self):
        """Wait for zombies after self.timeout seconds of inactivity.

            May be extended, do not override.
            """
        self.collect_children()

    def service_actions(self):
        """Collect the zombie child processes regularly in the ForkingMixIn.

            service_actions is called in the BaseServer's serve_forever loop.
            """
        self.collect_children()

    def process_request(self, request, client_address):
        """Fork a new subprocess to process the request."""
        pid = os.fork()
        if pid:
            if self.active_children is None:
                self.active_children = set()
            self.active_children.add(pid)
            self.close_request(request)
            return
        else:
            status = 1
            try:
                self.finish_request(request, client_address)
                status = 0
            except Exception:
                self.handle_error(request, client_address)
            finally:
                try:
                    self.shutdown_request(request)
                finally:
                    os._exit(status)

    def server_close(self):
        super().server_close()
        self.collect_children(blocking=self.block_on_close)