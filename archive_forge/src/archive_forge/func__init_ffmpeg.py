import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _init_ffmpeg():
    ffmpeg_vers = _get_ffmpeg_versions()
    ext = _find_ffmpeg_extension(ffmpeg_vers)
    ext.init()
    if ext.get_log_level() > 8:
        ext.set_log_level(8)
    return ext