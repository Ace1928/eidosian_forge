import json
import os
import pathlib
import shutil
from pathlib import Path
from typing import Text
from jupyter_server.serverapp import ServerApp
from pytest import fixture
from tornado.httpserver import HTTPRequest
from tornado.httputil import HTTPServerRequest
from tornado.queues import Queue
from tornado.web import Application
from jupyter_lsp import LanguageServerManager
from jupyter_lsp.constants import APP_CONFIG_D_SECTIONS
from jupyter_lsp.handlers import LanguageServersHandler, LanguageServerWebSocketHandler
@fixture(params=sorted(APP_CONFIG_D_SECTIONS))
def app_config_d(request, tmp_path, monkeypatch) -> pathlib.Path:
    conf_d = tmp_path / f'jupyter{request.param}config.d'
    conf_d.mkdir()
    monkeypatch.setenv('JUPYTER_CONFIG_PATH', f'{tmp_path}')
    return conf_d