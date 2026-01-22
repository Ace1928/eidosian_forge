import base64
import logging
import cloudpickle
from threading import Thread
from typing import Any, Optional, Tuple, Dict, List
import requests  # type: ignore
from fugue.rpc.base import RPCClient, RPCServer
from triad.utils.convert import to_timedelta
from werkzeug.serving import make_server
from flask import Flask, request
class FlaskRPCServer(RPCServer):
    """Flask RPC server that can be used in a distributed environment.
    It's required to set ``fugue.rpc.flask_server.host`` and
    ``fugue.rpc.flask_server.port``. If ``fugue.rpc.flask_server.timeout``
    is not set, then the client could hang until the connection is closed.

    :param conf: |FugueConfig|

    .. admonition:: Examples

        .. code-block:: python

            conf = {
                "fugue.rpc.server": "fugue.rpc.flask.FlaskRPCServer",
                "fugue.rpc.flask_server.host": "127.0.0.1",
                "fugue.rpc.flask_server.port": "1234",
                "fugue.rpc.flask_server.timeout": "2 sec",
            }

            with make_rpc_server(conf).start() as server:
                server...
    """

    class _Thread(Thread):

        def __init__(self, app: Flask, host: str, port: int):
            super().__init__()
            self._srv = make_server(host, port, app)
            self._ctx = app.app_context()
            self._ctx.push()

        def run(self) -> None:
            self._srv.serve_forever()

        def shutdown(self) -> None:
            self._srv.shutdown()

    def __init__(self, conf: Any):
        super().__init__(conf)
        self._host = conf.get_or_throw('fugue.rpc.flask_server.host', str)
        self._port = conf.get_or_throw('fugue.rpc.flask_server.port', int)
        timeout = conf.get_or_none('fugue.rpc.flask_server.timeout', object)
        self._timeout_sec = -1.0 if timeout is None else to_timedelta(timeout).total_seconds()
        self._server: Optional[FlaskRPCServer._Thread] = None

    def make_client(self, handler: Any) -> RPCClient:
        """Add ``handler`` and correspondent :class:`~.FlaskRPCClient`

        :param handler: |RPCHandlerLikeObject|
        :return: the flask RPC client that can be distributed
        """
        key = self.register(handler)
        return FlaskRPCClient(key, self._host, self._port, self._timeout_sec)

    def start_server(self) -> None:
        """Start Flask RPC server"""
        app = Flask('FlaskRPCServer')
        app.route('/invoke', methods=['POST'])(self._invoke)
        self._server = FlaskRPCServer._Thread(app, self._host, self._port)
        self._server.start()

    def stop_server(self) -> None:
        """Stop Flask RPC server"""
        if self._server is not None:
            self._server.shutdown()
            self._server.join()

    def _invoke(self) -> str:
        key = str(request.form.get('key'))
        args, kwargs = _decode(str(request.form.get('value')))
        return _encode(self.invoke(key, *args, **kwargs))