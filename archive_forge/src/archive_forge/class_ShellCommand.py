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
class ShellCommand(object):

    def __init__(self, command, wait=True, fail_fast=False, cwd=None):
        self.exit_code = 0
        self.command = command
        self.log_fp = StringIO()
        self.wait = wait
        self.fail_fast = fail_fast
        self.run(cwd=cwd)

    def run(self, cwd=None):
        boto.log.info('running:%s' % self.command)
        self.process = subprocess.Popen(self.command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        if self.wait:
            while self.process.poll() is None:
                time.sleep(1)
                t = self.process.communicate()
                self.log_fp.write(t[0])
                self.log_fp.write(t[1])
            boto.log.info(self.log_fp.getvalue())
            self.exit_code = self.process.returncode
            if self.fail_fast and self.exit_code != 0:
                raise Exception('Command ' + self.command + ' failed with status ' + self.exit_code)
            return self.exit_code

    def setReadOnly(self, value):
        raise AttributeError

    def getStatus(self):
        return self.exit_code
    status = property(getStatus, setReadOnly, None, 'The exit code for the command')

    def getOutput(self):
        return self.log_fp.getvalue()
    output = property(getOutput, setReadOnly, None, 'The STDIN and STDERR output of the command')