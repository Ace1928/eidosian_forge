from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.core.resource import custom_printer_base as cp
def GetContainer(container: container_resource.Container, labels: Mapping[str, str], dependencies: Sequence[str], is_primary: bool) -> cp.Table:
    limits = GetLimits(container)
    return cp.Labeled([('Image', container.image), ('Command', ' '.join(container.command)), ('Args', ' '.join(container.args)), ('Port', ' '.join((str(p.containerPort) for p in container.ports))), ('Memory', limits['memory']), ('CPU', limits['cpu']), ('Env vars', GetUserEnvironmentVariables(container)), ('Volume Mounts', GetVolumeMounts(container)), ('Secrets', GetSecrets(container)), ('Config Maps', GetConfigMaps(container)), ('Startup Probe', k8s_util.GetStartupProbe(container, labels, is_primary)), ('Liveness Probe', k8s_util.GetLivenessProbe(container)), ('Container Dependencies', ', '.join(dependencies))])