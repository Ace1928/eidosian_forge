import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import numpy as np
import torch
from PIL import Image
from .utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from .vision import VisionDataset
class KMNIST(MNIST):
    """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``KMNIST/raw/train-images-idx3-ubyte``
            and  ``KMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    mirrors = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/']
    resources = [('train-images-idx3-ubyte.gz', 'bdb82020997e1d708af4cf47b453dcf7'), ('train-labels-idx1-ubyte.gz', 'e144d726b3acfaa3e44228e80efcd344'), ('t10k-images-idx3-ubyte.gz', '5c965bf0a639b31b8f53240b1b52f4d7'), ('t10k-labels-idx1-ubyte.gz', '7320c461ea6c1c855c0b718fb2a4b134')]
    classes = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']