from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import base64
import errno
import io
import json
import logging
import os
import subprocess
from containerregistry.client import docker_name
import httplib2
from oauth2client import client as oauth2client
import six
class SchemeProvider(Provider):
    """Implementation for providing a challenge response credential."""

    def __init__(self, scheme):
        self._scheme = scheme

    @property
    @abc.abstractmethod
    def suffix(self):
        """Returns the authentication payload to follow the auth scheme."""

    def Get(self):
        """Gets the credential in a form suitable for an Authorization header."""
        return u'%s %s' % (self._scheme, self.suffix)