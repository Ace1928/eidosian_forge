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
def _process_dataset(dataset, output_dir, here, use_test_data=False, force=False):
    """Process each download spec in datasets.yml

    Typically each dataset list entry in the yml has
    "files" and "url" and "title" keys/values to show
    local files that must be present / extracted from
    a decompression of contents downloaded from the url.

    If a url endswith '/', then all files given
    are assumed to be added to the url pattern at the
    end
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with DirectoryContext(output_dir):
        requires_download = False
        for f in dataset.get('files', []):
            if not os.path.exists(f):
                requires_download = True
                break
        if force is False and (not requires_download):
            print('Skipping {0}'.format(dataset['title']))
            return
        url = dataset['url']
        title_fmt = dataset['title'] + ' {} of {}'
        if url.endswith('/'):
            urls = [url + f for f in dataset['files']]
            output_paths = [os.path.join(here, DATA_DIR, fname) for fname in dataset['files']]
            unpacked = ['.'.join(output_path.split('.')[:-2 if output_path.endswith('gz') else -1]) + '*' for output_path in output_paths]
        else:
            urls = [url]
            output_paths = [os.path.split(url)[1]]
            unpacked = dataset['files']
            if not isinstance(unpacked, (tuple, list)):
                unpacked = [unpacked]
        zipped = zip(urls, output_paths, unpacked)
        for idx, (url, output_path, unpack) in enumerate(zipped):
            running_title = title_fmt.format(idx + 1, len(urls))
            if force is False and (glob.glob(unpack) or os.path.exists(unpack.replace('*', ''))):
                print('Skipping {0}'.format(running_title))
                continue
            test = os.path.join(output_dir, DATA_STUBS_DIR, unpack)
            if use_test_data and os.path.exists(test):
                target = os.path.join(output_dir, unpack)
                print("Copying test data file '{0}' to '{1}'".format(test, target))
                shutil.copyfile(test, target)
                continue
            elif use_test_data and (not os.path.exists(test)):
                print('No test file found for: {}. Using regular file instead'.format(test))
            _url_to_binary_write(url, output_path, running_title)
            _extract_downloaded_archive(output_path)
    if requests is None:
        print('this download script requires the requests module: conda install requests')
        sys.exit(1)