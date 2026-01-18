from tornado import ioloop, web
from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import JupyterHandler
Shut down the server.