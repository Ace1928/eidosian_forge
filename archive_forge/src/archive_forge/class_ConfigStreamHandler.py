import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
class ConfigStreamHandler(StreamRequestHandler):
    """
        Handler for a logging configuration request.

        It expects a completely new logging configuration and uses fileConfig
        to install it.
        """

    def handle(self):
        """
            Handle a request.

            Each request is expected to be a 4-byte length, packed using
            struct.pack(">L", n), followed by the config file.
            Uses fileConfig() to do the grunt work.
            """
        try:
            conn = self.connection
            chunk = conn.recv(4)
            if len(chunk) == 4:
                slen = struct.unpack('>L', chunk)[0]
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk = chunk + conn.recv(slen - len(chunk))
                if self.server.verify is not None:
                    chunk = self.server.verify(chunk)
                if chunk is not None:
                    chunk = chunk.decode('utf-8')
                    try:
                        import json
                        d = json.loads(chunk)
                        assert isinstance(d, dict)
                        dictConfig(d)
                    except Exception:
                        file = io.StringIO(chunk)
                        try:
                            fileConfig(file)
                        except Exception:
                            traceback.print_exc()
                if self.server.ready:
                    self.server.ready.set()
        except OSError as e:
            if e.errno != RESET_ERROR:
                raise