from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Optional
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _RemovePath(self, domain_res: runapps_v1alpha1_messages.Resource, path: str):
    for route in domain_res.bindings:
        cfg = encoding.MessageToDict(route.config)
        paths = cfg.get('paths')
        for route_path in paths:
            if route_path == path:
                paths.remove(route_path)
                break
        if paths:
            route.config = encoding.DictToMessage(cfg, runapps_v1alpha1_messages.Binding.ConfigValue)
        else:
            domain_res.bindings.remove(route)