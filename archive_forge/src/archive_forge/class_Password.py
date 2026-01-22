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
class Password(object):
    """
    Password object that stores itself as hashed.
    Hash defaults to SHA512 if available, MD5 otherwise.
    """
    hashfunc = _hashfn

    def __init__(self, str=None, hashfunc=None):
        """
        Load the string from an initial value, this should be the
        raw hashed password.
        """
        self.str = str
        if hashfunc:
            self.hashfunc = hashfunc

    def set(self, value):
        if not isinstance(value, bytes):
            value = value.encode('utf-8')
        self.str = self.hashfunc(value).hexdigest()

    def __str__(self):
        return str(self.str)

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, bytes):
            other = other.encode('utf-8')
        return str(self.hashfunc(other).hexdigest()) == str(self.str)

    def __len__(self):
        if self.str:
            return len(self.str)
        else:
            return 0