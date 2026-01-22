from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def RefTransform(ref):
    if 'resolver' in ref:
        ref['resolver'] = ref.pop('resolver').upper()
    ParamDictTransform(ref.get('params', []))