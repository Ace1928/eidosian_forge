from __future__ import annotations
import csv
import hashlib
import os.path
import re
import stat
import time
from io import StringIO, TextIOWrapper
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from wheel.cli import WheelError
from wheel.util import log, urlsafe_b64decode, urlsafe_b64encode
A ZipFile derivative class that also reads SHA-256 hashes from
    .dist-info/RECORD and checks any read files against those.
    