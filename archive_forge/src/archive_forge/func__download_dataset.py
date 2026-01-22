import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
def _download_dataset(data_path: DataPath, dest: Path, attributes: Optional[typing.Iterable[str]], block_size: int, force: bool=False) -> None:
    """Downloads the dataset at ``data_path`` to ``dest``, optionally downloading
    only requested attributes. If ``attributes`` is not provided, every attribute
    will be requested.

    If any of the attributes of the remote dataset are already downloaded locally,
    they will not be overwritten unless ``force`` is True.
    """
    url_safe_datapath = urllib.parse.quote(str(data_path))
    s3_url = f'{S3_URL}/{url_safe_datapath}'
    if attributes is not None or dest.exists():
        _download_partial(s3_url, dest=dest, attributes=attributes, overwrite=force, block_size=block_size)
    else:
        _download_full(s3_url, dest=dest)