import glob
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pyarrow as pa
import datasets
import datasets.config
import datasets.data_files
from datasets.naming import filenames_for_dataset_split
def _get_modification_time(cached_directory_path):
    return Path(cached_directory_path).stat().st_mtime