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
def update_dme(username, password, dme_id, ip_address):
    """
    Update your Dynamic DNS record with DNSMadeEasy.com
    """
    dme_url = 'https://www.dnsmadeeasy.com/servlet/updateip'
    dme_url += '?username=%s&password=%s&id=%s&ip=%s'
    s = urllib.request.urlopen(dme_url % (username, password, dme_id, ip_address))
    return s.read()