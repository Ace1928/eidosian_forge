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
def AddRepo(self, membership, repo, rg, cluster_type):
    key = _GetSourceKey(repo)
    self.repos[key][membership] = RepoResourceGroupPair(repo, rg, cluster_type)