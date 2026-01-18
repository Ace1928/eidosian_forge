from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible import context
import socket
import uuid
import logging
from datetime import datetime
from ansible.plugins.callback import CallbackBase

    Tasks and handler tasks are dealt with here
    