import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
def write_to_fd(fd, data):
    """Write given data to given file descriptor, doing any conversions needed"""
    if six.PY2:
        fd.write(data)
        return
    if isinstance(data, bytes):
        if hasattr(fd, 'mode') and 'b' in fd.mode or isinstance(fd, io.BytesIO):
            fd.write(data)
        elif hasattr(fd, 'buffer'):
            fd.buffer.write(data)
        else:
            fd.write(six.ensure_text(data))
    elif 'b' in fd.mode:
        fd.write(six.ensure_binary(data))
    else:
        fd.write(data)