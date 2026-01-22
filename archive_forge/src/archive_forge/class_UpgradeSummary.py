from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
class UpgradeSummary:
    """UpgradeSummary holds image upgrade information."""

    def __init__(self):
        self.upgrades = []

    def AddOccurrence(self, occ):
        self.upgrades.append(occ)