from collections import abc
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from glance.common import exception
from glance.domain import proxy
from glance import policies
class ImageTarget(abc.Mapping):
    SENTINEL = object()

    def __init__(self, target):
        """Initialize the object

        :param target: Object being targeted
        """
        self.target = target
        self._target_keys = [k for k in dir(proxy.Image) if not k.startswith('__') if not k == 'locations' if not callable(getattr(proxy.Image, k))]

    def __getitem__(self, key):
        """Return the value of 'key' from the target.

        If the target has the attribute 'key', return it.

        :param key: value to retrieve
        """
        key = self.key_transforms(key)
        value = getattr(self.target, key, self.SENTINEL)
        if value is self.SENTINEL:
            extra_properties = getattr(self.target, 'extra_properties', None)
            if extra_properties is not None:
                value = extra_properties[key]
            else:
                value = None
        return value

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __len__(self):
        length = len(self._target_keys)
        length += len(getattr(self.target, 'extra_properties', {}))
        return length

    def __iter__(self):
        for key in self._target_keys:
            yield key
        for key in getattr(self.target, 'extra_properties', {}).keys():
            yield key
        for alias in ['project_id']:
            yield alias

    def key_transforms(self, key):
        transforms = {'id': 'image_id', 'project_id': 'owner', 'member_id': 'member'}
        return transforms.get(key, key)