import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _package_to_columns(self, pkg):
    """
        Given a package, return a list of values describing that
        package, one for each column in ``self.COLUMNS``.
        """
    row = []
    for column_index, column_name in enumerate(self.COLUMNS):
        if column_index == 0:
            row.append('')
        elif column_name == 'Identifier':
            row.append(pkg.id)
        elif column_name == 'Status':
            row.append(self._ds.status(pkg))
        else:
            attr = column_name.lower().replace(' ', '_')
            row.append(getattr(pkg, attr, 'n/a'))
    return row