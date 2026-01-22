import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
class PostRequest(Setting):
    name = 'post_request'
    section = 'Server Hooks'
    validator = validate_post_request
    type = callable

    def post_request(worker, req, environ, resp):
        pass
    default = staticmethod(post_request)
    desc = '        Called after a worker processes the request.\n\n        The callable needs to accept two instance variables for the Worker and\n        the Request.\n        '