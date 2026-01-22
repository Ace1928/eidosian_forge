import os
from tornado import web
from jupyter_server.utils import url_path_join, url_escape
from nbclient.util import ensure_async
from .utils import get_server_root_dir
from .handler import BaseVoilaHandler
def allowed_content(content):
    if content['type'] in ['directory', 'notebook']:
        return True
    __, ext = os.path.splitext(content.get('path'))
    return ext in self.allowed_extensions