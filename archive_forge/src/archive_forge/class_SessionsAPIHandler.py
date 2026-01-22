import asyncio
import json
from jupyter_client.kernelspec import NoSuchKernel
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_path_join
from ...base.handlers import APIHandler
class SessionsAPIHandler(APIHandler):
    """A Sessions API handler."""
    auth_resource = AUTH_RESOURCE