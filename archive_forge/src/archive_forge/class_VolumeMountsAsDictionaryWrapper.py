from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
class VolumeMountsAsDictionaryWrapper(k8s_object.KeyValueListAsDictionaryWrapper):
    """Wraps a list of volume mounts in a mutable dict-like object.

  Additionally provides properties to access mounts that are mounting volumes
  of specific type in a mutable dict-like object.
  """

    def __init__(self, volumes, mounts_to_wrap, mount_class):
        """Wraps a list of volume mounts in a mutable dict-like object.

    Args:
      volumes: associated VolumesAsDictionaryWrapper obj
      mounts_to_wrap: list[VolumeMount], list of mounts to treat as a dict.
      mount_class: type of the underlying VolumeMount objects.
    """
        super(VolumeMountsAsDictionaryWrapper, self).__init__(mounts_to_wrap, mount_class, key_field='mountPath', value_field='name')
        self._volumes = volumes

    @property
    def secrets(self):
        """Mutable dict-like object for mounts whose volumes have a secret source type."""
        return k8s_object.KeyValueListAsDictionaryWrapper(self._m, self._item_class, key_field=self._key_field, value_field=self._value_field, filter_func=lambda mount: mount.name in self._volumes.secrets)

    @property
    def config_maps(self):
        """Mutable dict-like object for mounts whose volumes have a config map source type."""
        return k8s_object.KeyValueListAsDictionaryWrapper(self._m, self._item_class, key_field=self._key_field, value_field=self._value_field, filter_func=lambda mount: mount.name in self._volumes.config_maps)