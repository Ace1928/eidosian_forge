from __future__ import annotations
import logging # isort:skip
import os
from pathlib import Path
from tornado.web import HTTPError, StaticFileHandler
from ...core.types import PathLike
 Serve static files from multiple, dynamically defined locations.

