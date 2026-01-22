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
def _make_ds_structure(self) -> None:
    """move videos from
        split_folder/
            ├── clip1.avi
            ├── clip2.avi

        to the correct format as described below:
        split_folder/
            ├── class1
            │   ├── clip1.avi

        """
    annotation_path = path.join(self.root, 'annotations')
    if not check_integrity(path.join(annotation_path, f'{self.split}.csv')):
        download_url(self._ANNOTATION_URLS[self.num_classes].format(split=self.split), annotation_path)
    annotations = path.join(annotation_path, f'{self.split}.csv')
    file_fmtstr = '{ytid}_{start:06}_{end:06}.mp4'
    with open(annotations) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            f = file_fmtstr.format(ytid=row['youtube_id'], start=int(row['time_start']), end=int(row['time_end']))
            label = row['label'].replace(' ', '_').replace("'", '').replace('(', '').replace(')', '')
            os.makedirs(path.join(self.split_folder, label), exist_ok=True)
            downloaded_file = path.join(self.split_folder, f)
            if path.isfile(downloaded_file):
                os.replace(downloaded_file, path.join(self.split_folder, label, f))