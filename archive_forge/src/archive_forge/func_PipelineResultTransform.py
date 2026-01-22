from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def PipelineResultTransform(pipeline_result):
    if 'value' in pipeline_result:
        pipeline_result['value'] = ResultValueTransform(pipeline_result['value'])