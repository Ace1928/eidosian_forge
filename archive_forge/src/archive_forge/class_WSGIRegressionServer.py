import time
from paste.httpserver import WSGIServer
class WSGIRegressionServer(WSGIServer):
    """
    A threaded WSGIServer for use in regression testing.  To use this
    module, call serve(application, regression=True), and then call
    server.accept() to let it handle one request.  When finished, use
    server.stop() to shutdown the server. Note that all pending requests
    are processed before the server shuts down.
    """
    defaulttimeout = 10

    def __init__(self, *args, **kwargs):
        WSGIServer.__init__(self, *args, **kwargs)
        self.stopping = []
        self.pending = []
        self.timeout = self.defaulttimeout
        self.socket.settimeout(2)

    def serve_forever(self):
        from threading import Thread
        thread = Thread(target=self.serve_pending)
        thread.start()

    def reset_expires(self):
        if self.timeout:
            self.expires = time.time() + self.timeout

    def close_request(self, *args, **kwargs):
        WSGIServer.close_request(self, *args, **kwargs)
        self.pending.pop()
        self.reset_expires()

    def serve_pending(self):
        self.reset_expires()
        while not self.stopping or self.pending:
            now = time.time()
            if now > self.expires and self.timeout:
                print('\nWARNING: WSGIRegressionServer timeout exceeded\n')
                break
            if self.pending:
                self.handle_request()
            time.sleep(0.1)

    def stop(self):
        """ stop the server (called from tester's thread) """
        self.stopping.append(True)

    def accept(self, count=1):
        """ accept another request (called from tester's thread) """
        assert not self.stopping
        [self.pending.append(True) for x in range(count)]