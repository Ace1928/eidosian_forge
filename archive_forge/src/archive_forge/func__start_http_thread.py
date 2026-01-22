import os
import sys
import time
import warnings
import contextlib
import portend
def _start_http_thread(self):
    """HTTP servers MUST be running in new threads, so that the
        main thread persists to receive KeyboardInterrupt's. If an
        exception is raised in the httpserver's thread then it's
        trapped here, and the bus (and therefore our httpserver)
        are shut down.
        """
    try:
        self.httpserver.start()
    except KeyboardInterrupt:
        self.bus.log('<Ctrl-C> hit: shutting down HTTP server')
        self.interrupt = sys.exc_info()[1]
        self.bus.exit()
    except SystemExit:
        self.bus.log('SystemExit raised: shutting down HTTP server')
        self.interrupt = sys.exc_info()[1]
        self.bus.exit()
        raise
    except Exception:
        self.interrupt = sys.exc_info()[1]
        self.bus.log('Error in HTTP server: shutting down', traceback=True, level=40)
        self.bus.exit()
        raise