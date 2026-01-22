from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.kuberun import kubernetesobject
from googlecloudsdk.api_lib.kuberun import structuredout
class VolumeMounts:
    """Represents the volume mounts in a revision.spec.container."""

    def __init__(self, volumes, volume_mounts):
        self._volumes = volumes
        self._volume_mounts = volume_mounts

    @property
    def secrets(self):
        return {mount['mountPath']: mount['name'] for mount in self._volume_mounts if mount['name'] in self._volumes.secrets}

    @property
    def config_maps(self):
        return {mount['mountPath']: mount['name'] for mount in self._volume_mounts if mount['name'] in self._volumes.config_maps}