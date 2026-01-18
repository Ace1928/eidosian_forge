import os
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url
from .vision import VisionDataset
Download and extract the tarball, and download each individual photo.