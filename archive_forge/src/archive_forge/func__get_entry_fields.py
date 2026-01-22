import io
import time
from urllib.request import urlopen
from urllib.parse import quote
from typing import Dict, List
from Bio._utils import function_with_previous
def _get_entry_fields(db):
    return _get_fields(_BASE_URL + f'/entry/{db}?fields')