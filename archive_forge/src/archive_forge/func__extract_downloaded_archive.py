from __future__ import print_function, absolute_import, division
from . import __version__
from .report import report
import os
import importlib
import inspect
import argparse
import distutils.dir_util
import shutil
from collections import OrderedDict
import glob
import sys
import tarfile
import time
import zipfile
import yaml
def _extract_downloaded_archive(output_path):
    """Extract a local archive, e.g. zip or tar, then
    delete the archive"""
    if output_path.endswith('tar.gz'):
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith('tar'):
        with tarfile.open(output_path, 'r:') as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith('tar.bz2'):
        with tarfile.open(output_path, 'r:bz2') as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith('zip'):
        with zipfile.ZipFile(output_path, 'r') as zipf:
            zipf.extractall()
        os.remove(output_path)