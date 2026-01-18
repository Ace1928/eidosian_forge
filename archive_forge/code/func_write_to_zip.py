import argparse
import glob
import os
from pathlib import Path
from zipfile import ZipFile
def write_to_zip(file_path, strip_file_path, zf, prepend_str=''):
    stripped_file_path = prepend_str + remove_prefix(file_path, strip_file_dir + '/')
    path = Path(stripped_file_path)
    if path.name in DENY_LIST:
        return
    zf.write(file_path, stripped_file_path)