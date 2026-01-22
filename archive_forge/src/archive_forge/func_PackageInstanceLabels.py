from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def PackageInstanceLabels(labels, messages):
    return encoding.DictToAdditionalPropertyMessage(labels, messages.Instance.LabelsValue, sort_items=True)