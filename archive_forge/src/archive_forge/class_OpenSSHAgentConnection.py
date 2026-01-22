import os.path
import time
class OpenSSHAgentConnection:

    def __init__(self):
        while True:
            try:
                self._pipe = os.open(PIPE_NAME, os.O_RDWR | os.O_BINARY)
            except OSError as e:
                if e.errno != 22:
                    raise
            else:
                break
            time.sleep(0.1)

    def send(self, data):
        return os.write(self._pipe, data)

    def recv(self, n):
        return os.read(self._pipe, n)

    def close(self):
        return os.close(self._pipe)