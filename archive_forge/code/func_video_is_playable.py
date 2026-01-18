from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
import aiofiles
import httpx
import numpy as np
from gradio_client import utils as client_utils
from PIL import Image, ImageOps, PngImagePlugin
from gradio import utils, wasm_utils
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.utils import abspath, get_upload_folder, is_in_or_equal
def video_is_playable(video_filepath: str) -> bool:
    """Determines if a video is playable in the browser.

    A video is playable if it has a playable container and codec.
        .mp4 -> h264
        .webm -> vp9
        .ogg -> theora
    """
    from ffmpy import FFprobe, FFRuntimeError
    try:
        container = Path(video_filepath).suffix.lower()
        probe = FFprobe(global_options='-show_format -show_streams -select_streams v -print_format json', inputs={video_filepath: None})
        output = probe.run(stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = json.loads(output[0])
        video_codec = output['streams'][0]['codec_name']
        return (container, video_codec) in [('.mp4', 'h264'), ('.ogg', 'theora'), ('.webm', 'vp9')]
    except (FFRuntimeError, IndexError, KeyError):
        return True