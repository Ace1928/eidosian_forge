from __future__ import print_function
import logging
import socket
import sys
from six.moves import BaseHTTPServer
from six.moves import http_client
from six.moves import input
from six.moves import urllib
from oauth2client import _helpers
from oauth2client import client
def _CreateArgumentParser():
    try:
        import argparse
    except ImportError:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--auth_host_name', default='localhost', help='Hostname when running a local web server.')
    parser.add_argument('--noauth_local_webserver', action='store_true', default=False, help='Do not run a local web server.')
    parser.add_argument('--auth_host_port', default=[8080, 8090], type=int, nargs='*', help='Port web server should listen on.')
    parser.add_argument('--logging_level', default='ERROR', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level of detail.')
    return parser