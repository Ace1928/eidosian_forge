import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command

    Find the path to Qt/lib folder inside the wheel.
    