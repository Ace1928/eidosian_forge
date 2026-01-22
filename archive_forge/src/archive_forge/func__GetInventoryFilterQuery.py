from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
import zlib
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_projector
def _GetInventoryFilterQuery(self, args):
    query_list = []

    def _AppendQuery(query):
        query_list.append('({})'.format(query))
    if args.inventory_filter:
        _AppendQuery(args.inventory_filter)
    if args.os_shortname:
        _AppendQuery('ShortName=' + args.os_shortname)
    if args.os_version:
        _AppendQuery('Version=' + args.os_version)
    if args.kernel_version:
        _AppendQuery('KernelVersion=' + args.kernel_version)
    installed_packages_query_prefixes = ['InstalledPackages.' + package_manager + '[].' for package_manager in self._REGULAR_PACKAGE_MANAGERS]
    if args.package_version:
        if not args.package_name:
            raise exceptions.InvalidArgumentException('--package-version', 'package version must be specified together with a package name. e.g. --package-name google-cloud-sdk --package-version 235.0.0-0')
        else:
            package_name = "['{}']".format(args.package_name)
            _AppendQuery(' OR '.join(['({})'.format(prefix + package_name + '.Version=' + args.package_version) for prefix in installed_packages_query_prefixes]))
    elif args.package_name:
        _AppendQuery(' OR '.join(['({})'.format(prefix + 'Name=' + args.package_name) for prefix in installed_packages_query_prefixes]))
    return ' AND '.join(query_list)