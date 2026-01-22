import csv
import os
import time
import urllib
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple
from torch import Tensor
from .folder import find_classes, make_dataset
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .video_utils import VideoClips
from .vision import VisionDataset
def _dl_wrap(tarpath: str, videopath: str, line: str) -> None:
    download_and_extract_archive(line, tarpath, videopath)