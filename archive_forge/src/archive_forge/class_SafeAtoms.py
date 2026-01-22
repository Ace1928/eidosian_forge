import base64
import binascii
import json
import time
import logging
from logging.config import dictConfig
from logging.config import fileConfig
import os
import socket
import sys
import threading
import traceback
from gunicorn import util
class SafeAtoms(dict):

    def __init__(self, atoms):
        dict.__init__(self)
        for key, value in atoms.items():
            if isinstance(value, str):
                self[key] = value.replace('"', '\\"')
            else:
                self[key] = value

    def __getitem__(self, k):
        if k.startswith('{'):
            kl = k.lower()
            if kl in self:
                return super().__getitem__(kl)
            else:
                return '-'
        if k in self:
            return super().__getitem__(k)
        else:
            return '-'