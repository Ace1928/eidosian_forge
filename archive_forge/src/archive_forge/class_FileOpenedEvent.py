from __future__ import annotations
import logging
import os.path
import re
from dataclasses import dataclass, field
from typing import Optional
from watchdog.utils.patterns import match_any_paths
class FileOpenedEvent(FileSystemEvent):
    """File system event representing file close on the file system."""
    event_type = EVENT_TYPE_OPENED