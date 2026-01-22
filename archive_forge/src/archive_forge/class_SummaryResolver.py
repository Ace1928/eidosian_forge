from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
class SummaryResolver(object):
    """SummaryResolver is a base class for occurrence summary objects."""

    def resolve(self):
        """resolve is called after all records are added to the summary.

    In this function, aggregate data can be calculated for display.
    """
        pass