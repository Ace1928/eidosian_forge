import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
def _server_loop(self, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        s.settimeout(0.5)
        while self._run.value == 1:
            try:
                c_socket, addr = s.accept()
            except socket.timeout:
                continue
            t = threading.Thread(target=self._process_request, args=(c_socket,), daemon=True)
            t.start()