import os
from tornado import web
from jupyter_server.utils import url_path_join, url_escape
from nbclient.util import ensure_async
from .utils import get_server_root_dir
from .handler import BaseVoilaHandler
Return the jinja template object for a given name