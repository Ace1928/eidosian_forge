from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def TaskResultTransform(task_result):
    _ConvertToUpperCase(task_result, 'type')
    for property_name in task_result.get('properties', []):
        PropertySpecTransform(task_result['properties'][property_name])