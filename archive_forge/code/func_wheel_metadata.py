import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import UnsupportedWheel
def wheel_metadata(source: ZipFile, dist_info_dir: str) -> Message:
    """Return the WHEEL metadata of an extracted wheel, if possible.
    Otherwise, raise UnsupportedWheel.
    """
    path = f'{dist_info_dir}/WHEEL'
    wheel_contents = read_wheel_metadata_file(source, path)
    try:
        wheel_text = wheel_contents.decode()
    except UnicodeDecodeError as e:
        raise UnsupportedWheel(f'error decoding {path!r}: {e!r}')
    return Parser().parsestr(wheel_text)