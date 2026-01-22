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
class BaseDescriber(base.DescribeCommand, BaseCommand):
    """Base class for the describe subcommands."""
    service = None

    @staticmethod
    def Args(parser, resource=None):
        BaseDescriber.AddArgs(parser, resource)

    @staticmethod
    def AddArgs(parser, resource=None):
        parser.add_argument('name', metavar='NAME', help='The name of the resource to fetch.')

    @property
    def method(self):
        return 'Get'

    def ScopeRequest(self, ref, request):
        """Adds a zone or region to the request object if necessary."""

    @abc.abstractmethod
    def CreateReference(self, args):
        pass

    def SetNameField(self, ref, request):
        """Sets the field in the request that corresponds to the object name."""
        name_field = self.service.GetMethodConfig(self.method).ordered_params[-1]
        setattr(request, name_field, ref.Name())

    def ComputeDynamicProperties(self, args, items):
        """Computes dynamic properties, which are not returned by GCE API."""
        _ = args
        return items

    def Run(self, args):
        """Yields JSON-serializable dicts of resources."""
        ref = self.CreateReference(args)
        get_request_class = self.service.GetRequestType(self.method)
        request = get_request_class(project=getattr(ref, 'project', self.project))
        self.SetNameField(ref, request)
        self.ScopeRequest(ref, request)
        get_request = (self.service, self.method, request)
        errors = []
        objects = request_helper.MakeRequests(requests=[get_request], http=self.http, batch_url=self.batch_url, errors=errors)
        resource_list = lister.ProcessResults(objects, field_selector=None)
        resource_list = list(self.ComputeDynamicProperties(args, resource_list))
        if errors:
            utils.RaiseToolException(errors, error_message='Could not fetch resource:')
        return resource_list[0]