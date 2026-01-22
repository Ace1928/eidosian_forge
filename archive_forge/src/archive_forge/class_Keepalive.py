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
class Keepalive(Setting):
    name = 'keepalive'
    section = 'Worker Processes'
    cli = ['--keep-alive']
    meta = 'INT'
    validator = validate_pos_int
    type = int
    default = 2
    desc = "        The number of seconds to wait for requests on a Keep-Alive connection.\n\n        Generally set in the 1-5 seconds range for servers with direct connection\n        to the client (e.g. when you don't have separate load balancer). When\n        Gunicorn is deployed behind a load balancer, it often makes sense to\n        set this to a higher value.\n\n        .. note::\n           ``sync`` worker does not support persistent connections and will\n           ignore this option.\n        "