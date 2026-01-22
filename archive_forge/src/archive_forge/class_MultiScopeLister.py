from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse  # pylint: disable=unused-import
import json
import textwrap
from apitools.base.py import base_api  # pylint: disable=unused-import
import enum
from googlecloudsdk.api_lib.compute import base_classes_resource_registry as resource_registry
from googlecloudsdk.api_lib.compute import client_adapter
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import resource_specs
from googlecloudsdk.api_lib.compute import scope_prompter
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
class MultiScopeLister(BaseLister):
    """Base class for listing global and regional resources."""

    @staticmethod
    def AddScopeArgs(parser, scopes):
        BaseLister.Args(parser)
        scope = parser.add_mutually_exclusive_group()
        if ScopeType.zonal_scope in scopes:
            scope.add_argument('--zones', metavar='ZONE', help='If provided, only zonal resources are shown. If arguments are provided, only resources from the given zones are shown.', type=arg_parsers.ArgList())
        if ScopeType.regional_scope in scopes:
            scope.add_argument('--regions', metavar='REGION', help='If provided, only regional resources are shown. If arguments are provided, only resources from the given regions are shown.', type=arg_parsers.ArgList())
        if ScopeType.global_scope in scopes:
            scope.add_argument('--global', action='store_true', help='If provided, only global resources are shown.', default=False)

    @abc.abstractproperty
    def global_service(self):
        """The service used to list global resources."""

    @abc.abstractproperty
    def regional_service(self):
        """The service used to list regional resources."""

    @abc.abstractproperty
    def zonal_service(self):
        """The service used to list regional resources."""

    @abc.abstractproperty
    def aggregation_service(self):
        """The service used to get aggregated list of resources."""

    def GetResources(self, args, errors):
        """Returns zonal, regional and/or global resources.

    Args:
      args: argparse.Namespace, Parsed arguments
      errors: list, Errors will be returned in this list

    Returns:
      Zonal, regional and/or global resources.
    """
        regions = getattr(args, 'regions', None)
        zones = getattr(args, 'zones', None)
        global_arg = getattr(args, 'global', None)
        no_scope_flags = not regions and (not zones) and (not global_arg)
        requests = []
        filter_expr = self.GetFilterExpr(args)
        if args.page_size is not None:
            max_results = min(args.page_size, constants.MAX_RESULTS_PER_PAGE)
        else:
            max_results = constants.MAX_RESULTS_PER_PAGE
        project = self.project
        if no_scope_flags and self.aggregation_service:
            requests.append((self.aggregation_service, 'AggregatedList', self.aggregation_service.GetRequestType('AggregatedList')(filter=filter_expr, maxResults=max_results, project=project)))
        elif regions is not None:
            region_names = set((self.CreateGlobalReference(region, resource_type='regions').Name() for region in regions))
            for region_name in sorted(region_names):
                requests.append((self.regional_service, 'List', self.regional_service.GetRequestType('List')(filter=filter_expr, maxResults=max_results, region=region_name, project=project)))
        elif zones is not None:
            zone_names = set((self.CreateGlobalReference(zone, resource_type='zones').Name() for zone in zones))
            for zone_name in sorted(zone_names):
                requests.append((self.zonal_service, 'List', self.zonal_service.GetRequestType('List')(filter=filter_expr, maxResults=max_results, zone=zone_name, project=project)))
        else:
            requests.append((self.global_service, 'List', self.global_service.GetRequestType('List')(filter=filter_expr, maxResults=max_results, project=project)))
        return request_helper.ListJson(requests=requests, http=self.http, batch_url=self.batch_url, errors=errors)