from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.ext import builtins
class RecurseState(object):

    def __init__(self):
        self.includes = {}
        self.excludes = {}
        self.aggregate_appinclude = appinfo.AppInclude()