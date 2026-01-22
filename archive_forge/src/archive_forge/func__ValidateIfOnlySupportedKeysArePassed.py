from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _ValidateIfOnlySupportedKeysArePassed(keys: List[str], supported_keys: List[str]):
    for key in keys:
        if key not in supported_keys:
            raise InvalidAutoscalingSettingsProvidedError('unsupported key: {key}, supported keys are: {supported_keys}'.format(key=key, supported_keys=supported_keys))