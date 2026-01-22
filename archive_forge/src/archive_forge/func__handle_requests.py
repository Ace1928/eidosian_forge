import threading
import socket
import select
def _handle_requests(self):
    for _ in range(self.requests_to_handle):
        sock = self._accept_connection()
        if not sock:
            break
        handler_result = self.handler(sock)
        self.handler_results.append(handler_result)
        sock.close()