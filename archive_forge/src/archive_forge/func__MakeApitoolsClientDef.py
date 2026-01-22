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
def _MakeApitoolsClientDef(root_package, api_name, api_version):
    """Makes an ApitoolsClientDef."""
    class_path = '.'.join([root_package, api_name, api_version])
    if api_name == 'admin' and api_version == 'v1':
        client_classpath = 'admin_v1_client.AdminDirectoryV1'
    else:
        client_classpath = '.'.join(['_'.join([api_name, api_version, 'client']), _CamelCase(api_name) + _CamelCase(api_version)])
    messages_modulepath = '_'.join([api_name, api_version, 'messages'])
    base_url = ''
    client_full_classpath = class_path + '.' + client_classpath
    try:
        client_classpath_def = _GetClientClassFromDef(client_full_classpath)
        base_url = client_classpath_def.BASE_URL
    except Exception:
        pass
    apitools_def = api_def.ApitoolsClientDef(class_path=class_path, client_classpath=client_classpath, messages_modulepath=messages_modulepath, base_url=base_url)
    return apitools_def