from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_printer
import six
def AggregateListNetworks(self, project_resource, product, limit=None):
    """Make a series of List Networks requests."""
    _ValidateProduct(product)
    if product == _PFORG:
        power_resource = 'powerNetworks'
        return self.AggregateYieldFromList(self.power_networks_service, project_resource, self.messages.MarketplacesolutionsProjectsLocationsPowerNetworksListRequest, power_resource, limit=limit)