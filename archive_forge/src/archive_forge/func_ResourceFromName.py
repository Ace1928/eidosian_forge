from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
def ResourceFromName(self, name):
    return self._registry.Parse(name, collection='genomics.projects.operations', params={'projectsId': genomics_util.GetProjectId()})