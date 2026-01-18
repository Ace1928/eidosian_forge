from __future__ import annotations
import sys, os
import configparser
import shutil
import typing as T
from glob import glob
from .wrap import (open_wrapdburl, WrapException, get_releases, get_releases_data,
from pathlib import Path
from .. import mesonlib, msubprojects
def update_db(options: 'argparse.Namespace') -> None:
    data = get_releases_data(options.allow_insecure)
    Path('subprojects').mkdir(exist_ok=True)
    with Path('subprojects/wrapdb.json').open('wb') as f:
        f.write(data)