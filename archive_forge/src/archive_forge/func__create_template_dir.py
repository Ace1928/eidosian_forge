import atexit
import json
import os
import shutil
import sys
import tempfile
from os import path as osp
from os.path import join as pjoin
from stat import S_IRGRP, S_IROTH, S_IRUSR
from tempfile import TemporaryDirectory
from unittest.mock import patch
import jupyter_core
import jupyterlab_server
from ipykernel.kernelspec import write_kernel_spec
from jupyter_server.serverapp import ServerApp
from jupyterlab_server.process_app import ProcessApp
from traitlets import default
def _create_template_dir():
    template_dir = tempfile.mkdtemp(prefix='mock_static')
    index_filepath = osp.join(template_dir, 'index.html')
    with open(index_filepath, 'w') as fid:
        fid.write('\n<!DOCTYPE HTML>\n<html>\n<head>\n    <meta charset="utf-8">\n    <title>{% block title %}Jupyter Lab Test{% endblock %}</title>\n    <meta http-equiv="X-UA-Compatible" content="IE=edge" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    {% block meta %}\n    {% endblock %}\n</head>\n<body>\n  <h1>JupyterLab Test Application</h1>\n  <div id="site">\n    {% block site %}\n    {% endblock site %}\n  </div>\n  {% block after_site %}\n  {% endblock after_site %}\n</body>\n</html>')
    return template_dir