from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def _MakeApiMap(root_package, api_config):
    """Converts a map of api_config into ApiDef.

  Args:
    root_package: str, root path of where generate api will reside.
    api_config: {api_name->api_version->{discovery,default,version,...}},
                description of each api.
  Returns:
    {api_name->api_version->ApiDef()}.

  Raises:
    NoDefaultApiError: if for some api with multiple versions
        default was not specified.
  """
    apis_map = {}
    apis_with_default = set()
    for api_name, api_version_config in six.iteritems(api_config):
        api_versions_map = apis_map.setdefault(api_name, {})
        has_default = False
        for api_version, api_config in six.iteritems(api_version_config):
            if api_config.get('discovery_doc'):
                apitools_client = _MakeApitoolsClientDef(root_package, api_name, api_version)
            else:
                apitools_client = None
            if api_config.get('gcloud_gapic_library'):
                gapic_client = _MakeGapicClientDef(root_package, api_name, api_version)
            else:
                gapic_client = None
            default = api_config.get('default', len(api_version_config) == 1)
            if has_default and default:
                raise NoDefaultApiError('Multiple default client versions found for [{}]!'.format(api_name))
            has_default = has_default or default
            enable_mtls = api_config.get('enable_mtls', True)
            mtls_endpoint_override = api_config.get('mtls_endpoint_override', '')
            api_versions_map[api_version] = api_def.APIDef(apitools_client, gapic_client, default, enable_mtls, mtls_endpoint_override)
        if has_default:
            apis_with_default.add(api_name)
    apis_without_default = set(apis_map.keys()).difference(apis_with_default)
    if apis_without_default:
        raise NoDefaultApiError('No default client versions found for [{0}]!'.format(', '.join(sorted(apis_without_default))))
    return apis_map