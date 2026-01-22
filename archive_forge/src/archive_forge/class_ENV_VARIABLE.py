import asyncio
import json
import os
import threading
import warnings
from enum import Enum
from functools import partial
from typing import Awaitable, Dict
import websockets
from jupyterlab_server.themes_handler import ThemesHandler
from markupsafe import Markup
from nbconvert.exporters.html import find_lab_theme
from .static_file_handler import TemplateStaticFileHandler
class ENV_VARIABLE(str, Enum):
    VOILA_PREHEAT = 'VOILA_PREHEAT'
    VOILA_KERNEL_ID = 'VOILA_KERNEL_ID'
    VOILA_BASE_URL = 'VOILA_BASE_URL'
    VOILA_SERVER_URL = 'VOILA_SERVER_URL'
    VOILA_APP_IP = 'VOILA_APP_IP'
    VOILA_APP_PORT = 'VOILA_APP_PORT'
    VOILA_WS_PROTOCOL = 'VOILA_WS_PROTOCOL'
    VOILA_WS_BASE_URL = 'VOILA_WS_BASE_URL'
    SERVER_NAME = 'SERVER_NAME'
    SERVER_PORT = 'SERVER_PORT'
    SCRIPT_NAME = 'SCRIPT_NAME'
    PATH_INFO = 'PATH_INFO'
    QUERY_STRING = 'QUERY_STRING'
    SERVER_SOFTWARE = 'SERVER_SOFTWARE'
    SERVER_PROTOCOL = 'SERVER_PROTOCOL'