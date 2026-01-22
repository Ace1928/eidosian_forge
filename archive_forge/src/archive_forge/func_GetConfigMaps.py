from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.core.resource import custom_printer_base as cp
def GetConfigMaps(container: container_resource.Container) -> cp.Table:
    """Returns a print mapping for env var and volume-mounted config maps."""
    config_maps = {}
    config_maps.update({k: _FormatConfigMapKeyRef(v) for k, v in container.env_vars.config_maps.items()})
    config_maps.update({k: _FormatConfigMapVolumeSource(v) for k, v in container.MountedVolumeJoin('config_maps').items()})
    return cp.Mapped(k8s_util.OrderByKey(config_maps))