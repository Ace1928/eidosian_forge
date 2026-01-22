from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def ApplyYaml(self, yaml_content: str):
    """Applies the application config from yaml file.

    Args:
      yaml_content: content of the yaml file.
    """
    app_dict = dict(yaml_content)
    name = _DEFAULT_APP_NAME
    if 'name' in app_dict:
        name = app_dict.pop('name')
    appconfig = runapps_v1alpha1_messages.Config(config=yaml.dump(yaml_content).encode('utf-8'))
    match_type_names = []
    vpc = False
    for r in app_dict.get('resources', {}):
        res_id = r.get('id', '')
        parts = res_id.split('/')
        if len(parts) != 2:
            continue
        match_type_names.append({'type': parts[0], 'name': parts[1]})
        if parts[0] == 'redis':
            vpc = True
    if vpc:
        match_type_names.append({'type': 'vpc', 'name': '*'})
    match_type_names.sort(key=lambda x: x['type'])
    all_types = map(lambda x: x['type'], match_type_names)
    validator.CheckApiEnablements(all_types)
    resource_stages = base.GetComponentTypesFromSelectors(selectors=match_type_names)
    stages_map = stages.ApplyStages(resource_stages)

    def StatusUpdate(tracker, operation, unused_status):
        self._UpdateDeploymentTracker(tracker, operation, stages_map)
        return
    with progress_tracker.StagedProgressTracker('Applying Configuration...', stages_map.values(), failure_message='Failed to apply configuration.') as tracker:
        self.ApplyAppConfig(tracker=tracker, tracker_update_func=StatusUpdate, appname=name, appconfig=appconfig, match_type_names=match_type_names)