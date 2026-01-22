import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _unpack_archive(filename, extract_dir):
    try:
        shutil.unpack_archive(filename, extract_dir)
    except shutil.ReadError:
        print('The archive format is not supported in your Python environment. Falling back to "tar" command...')
        try:
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.run(['tar', 'xf', filename, '-C', extract_dir], check=True)
        except subprocess.CalledProcessError:
            msg = 'Failed to extract the archive using "tar" command.'
            raise RuntimeError(msg)