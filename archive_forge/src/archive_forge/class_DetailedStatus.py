from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
class DetailedStatus:
    """DetailedStatus represent a detailed status for a repo."""

    def __init__(self, source='', commit='', status='', errors=None, clusters=None):
        self.source = source
        self.commit = commit
        self.status = status
        self.clusters = clusters
        self.errors = errors

    def EqualTo(self, result):
        return self.source == result.source and self.commit == result.commit and (self.status == result.status)