from wsgiref import simple_server
from oslo_serialization import jsonutils
from keystonemiddleware import auth_token
class EchoService(object):
    """Runs an instance of the echo app on init."""

    def __init__(self):
        conf = {'auth_protocol': 'http', 'admin_token': 'ADMIN'}
        app = auth_token.AuthProtocol(echo_app, conf)
        server = simple_server.make_server('', 8000, app)
        print('Serving on port 8000 (Ctrl+C to end)...')
        server.serve_forever()