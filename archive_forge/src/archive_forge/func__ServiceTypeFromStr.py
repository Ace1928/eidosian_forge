from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from typing import List, Optional
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def _ServiceTypeFromStr(s: str) -> ServiceType:
    """Converts string into service type."""
    types = {'backing': ServiceType.BACKING, 'ingress': ServiceType.INGRESS, 'workload': ServiceType.WORKLOAD}
    service_type = types.get(s.lower(), None)
    if service_type is None:
        raise exceptions.ArgumentError('Service type {} is not supported'.format(s))
    return service_type