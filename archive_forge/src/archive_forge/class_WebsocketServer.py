import sys
import struct
import ssl
from base64 import b64encode
from hashlib import sha1
import logging
from socket import error as SocketError
import errno
import threading
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from websocket_server.thread import WebsocketServerThread
class WebsocketServer(ThreadingMixIn, TCPServer, API):
    """
	A websocket server waiting for clients to connect.

    Args:
        port(int): Port to bind to
        host(str): Hostname or IP to listen for connections. By default 127.0.0.1
            is being used. To accept connections from any client, you should use
            0.0.0.0.
        loglevel: Logging level from logging module to use for logging. By default
            warnings and errors are being logged.

    Properties:
        clients(list): A list of connected clients. A client is a dictionary
            like below.
                {
                 'id'      : id,
                 'handler' : handler,
                 'address' : (addr, port)
                }
    """
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host='127.0.0.1', port=0, loglevel=logging.WARNING, key=None, cert=None):
        logger.setLevel(loglevel)
        TCPServer.__init__(self, (host, port), WebSocketHandler)
        self.host = host
        self.port = self.socket.getsockname()[1]
        self.key = key
        self.cert = cert
        self.clients = []
        self.id_counter = 0
        self.thread = None
        self._deny_clients = False

    def _run_forever(self, threaded):
        cls_name = self.__class__.__name__
        try:
            logger.info('Listening on port %d for clients..' % self.port)
            if threaded:
                self.daemon = True
                self.thread = WebsocketServerThread(target=super().serve_forever, daemon=True, logger=logger)
                logger.info(f'Starting {cls_name} on thread {self.thread.getName()}.')
                self.thread.start()
            else:
                self.thread = threading.current_thread()
                logger.info(f'Starting {cls_name} on main thread.')
                super().serve_forever()
        except KeyboardInterrupt:
            self.server_close()
            logger.info('Server terminated.')
        except Exception as e:
            logger.error(str(e), exc_info=True)
            sys.exit(1)

    def _message_received_(self, handler, msg):
        self.message_received(self.handler_to_client(handler), self, msg)

    def _ping_received_(self, handler, msg):
        handler.send_pong(msg)

    def _pong_received_(self, handler, msg):
        pass

    def _new_client_(self, handler):
        if self._deny_clients:
            status = self._deny_clients['status']
            reason = self._deny_clients['reason']
            handler.send_close(status, reason)
            self._terminate_client_handler(handler)
            return
        self.id_counter += 1
        client = {'id': self.id_counter, 'handler': handler, 'address': handler.client_address}
        self.clients.append(client)
        self.new_client(client, self)

    def _client_left_(self, handler):
        client = self.handler_to_client(handler)
        self.client_left(client, self)
        if client in self.clients:
            self.clients.remove(client)

    def _unicast(self, receiver_client, msg):
        receiver_client['handler'].send_message(msg)

    def _multicast(self, msg):
        for client in self.clients:
            self._unicast(client, msg)

    def handler_to_client(self, handler):
        for client in self.clients:
            if client['handler'] == handler:
                return client

    def _terminate_client_handler(self, handler):
        handler.keep_alive = False
        handler.finish()
        handler.connection.close()

    def _terminate_client_handlers(self):
        """
        Ensures request handler for each client is terminated correctly
        """
        for client in self.clients:
            self._terminate_client_handler(client['handler'])

    def _shutdown_gracefully(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
        """
        Send a CLOSE handshake to all connected clients before terminating server
        """
        self.keep_alive = False
        self._disconnect_clients_gracefully(status, reason)
        self.server_close()
        self.shutdown()

    def _shutdown_abruptly(self):
        """
        Terminate server without sending a CLOSE handshake
        """
        self.keep_alive = False
        self._disconnect_clients_abruptly()
        self.server_close()
        self.shutdown()

    def _disconnect_clients_gracefully(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
        """
        Terminate clients gracefully without shutting down the server
        """
        for client in self.clients:
            client['handler'].send_close(status, reason)
        self._terminate_client_handlers()

    def _disconnect_clients_abruptly(self):
        """
        Terminate clients abruptly (no CLOSE handshake) without shutting down the server
        """
        self._terminate_client_handlers()

    def _deny_new_connections(self, status, reason):
        self._deny_clients = {'status': status, 'reason': reason}

    def _allow_new_connections(self):
        self._deny_clients = False