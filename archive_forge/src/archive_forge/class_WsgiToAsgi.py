from io import BytesIO
from tempfile import SpooledTemporaryFile
from asgiref.sync import AsyncToSync, sync_to_async
class WsgiToAsgi:
    """
    Wraps a WSGI application to make it into an ASGI application.
    """

    def __init__(self, wsgi_application):
        self.wsgi_application = wsgi_application

    async def __call__(self, scope, receive, send):
        """
        ASGI application instantiation point.
        We return a new WsgiToAsgiInstance here with the WSGI app
        and the scope, ready to respond when it is __call__ed.
        """
        await WsgiToAsgiInstance(self.wsgi_application)(scope, receive, send)