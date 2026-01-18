import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
Truncate string values in a dictionary if they exceed max_length.

        :param dict: Dictionary with potentially large values
        :param max_length: Maximum allowed length of string values
        :return: Dictionary with truncated string values
        