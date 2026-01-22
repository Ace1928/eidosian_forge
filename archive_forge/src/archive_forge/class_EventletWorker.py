from functools import partial
import sys
from eventlet import hubs, greenthread
from eventlet.greenio import GreenSocket
import eventlet.wsgi
import greenlet
from gunicorn.workers.base_async import AsyncWorker
from gunicorn.sock import ssl_wrap_socket
class EventletWorker(AsyncWorker):

    def patch(self):
        hubs.use_hub()
        eventlet.monkey_patch()
        patch_sendfile()

    def is_already_handled(self, respiter):
        if getattr(EVENTLET_WSGI_LOCAL, 'already_handled', None):
            raise StopIteration()
        if respiter == EVENTLET_ALREADY_HANDLED:
            raise StopIteration()
        return super().is_already_handled(respiter)

    def init_process(self):
        self.patch()
        super().init_process()

    def handle_quit(self, sig, frame):
        eventlet.spawn(super().handle_quit, sig, frame)

    def handle_usr1(self, sig, frame):
        eventlet.spawn(super().handle_usr1, sig, frame)

    def timeout_ctx(self):
        return eventlet.Timeout(self.cfg.keepalive or None, False)

    def handle(self, listener, client, addr):
        if self.cfg.is_ssl:
            client = ssl_wrap_socket(client, self.cfg)
        super().handle(listener, client, addr)

    def run(self):
        acceptors = []
        for sock in self.sockets:
            gsock = GreenSocket(sock)
            gsock.setblocking(1)
            hfun = partial(self.handle, gsock)
            acceptor = eventlet.spawn(_eventlet_serve, gsock, hfun, self.worker_connections)
            acceptors.append(acceptor)
            eventlet.sleep(0.0)
        while self.alive:
            self.notify()
            eventlet.sleep(1.0)
        self.notify()
        t = None
        try:
            with eventlet.Timeout(self.cfg.graceful_timeout) as t:
                for a in acceptors:
                    a.kill(eventlet.StopServe())
                for a in acceptors:
                    a.wait()
        except eventlet.Timeout as te:
            if te != t:
                raise
            for a in acceptors:
                a.kill()