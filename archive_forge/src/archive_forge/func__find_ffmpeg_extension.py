import importlib
import logging
import os
import types
from pathlib import Path
import torch
def _find_ffmpeg_extension(ffmpeg_vers):
    for ffmpeg_ver in ffmpeg_vers:
        _LG.debug('Loading FFmpeg%s', ffmpeg_ver)
        try:
            ext = _find_versionsed_ffmpeg_extension(ffmpeg_ver)
            _LG.debug('Successfully loaded FFmpeg%s', ffmpeg_ver)
            return ext
        except Exception:
            _LG.debug('Failed to load FFmpeg%s extension.', ffmpeg_ver, exc_info=True)
            continue
    raise ImportError(f'Failed to intialize FFmpeg extension. Tried versions: {ffmpeg_vers}. Enable DEBUG logging to see more details about the error.')