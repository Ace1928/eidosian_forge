import argparse
import warnings
from http.cookies import SimpleCookie
from shlex import split
from urllib.parse import urlparse
from w3lib.http import basic_auth_header
Convert a cURL command syntax to Request kwargs.

    :param str curl_command: string containing the curl command
    :param bool ignore_unknown_options: If true, only a warning is emitted when
                                        cURL options are unknown. Otherwise
                                        raises an error. (default: True)
    :return: dictionary of Request kwargs
    