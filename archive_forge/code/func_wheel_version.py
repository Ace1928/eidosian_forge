import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import UnsupportedWheel
def wheel_version(wheel_data: Message) -> Tuple[int, ...]:
    """Given WHEEL metadata, return the parsed Wheel-Version.
    Otherwise, raise UnsupportedWheel.
    """
    version_text = wheel_data['Wheel-Version']
    if version_text is None:
        raise UnsupportedWheel('WHEEL is missing Wheel-Version')
    version = version_text.strip()
    try:
        return tuple(map(int, version.split('.')))
    except ValueError:
        raise UnsupportedWheel(f'invalid Wheel-Version: {version!r}')