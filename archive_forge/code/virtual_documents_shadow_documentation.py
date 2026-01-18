from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import List
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from .manager import lsp_message_listener
from .paths import file_uri_to_path, is_relative
from .types import LanguageServerManagerAPI
Intercept a message with document contents creating a shadow file for it.

        Only create the shadow file if the URI matches the virtual documents URI.
        Returns the path on filesystem where the content was stored.
        