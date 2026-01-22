from zunclient.common import base
class AvailabilityZoneManager(base.Manager):
    resource_class = AvailabilityZone

    @staticmethod
    def _path():
        return '/v1/availability_zones'

    def list(self, **kwargs):
        return self._list(self._path(), 'availability_zones', qparams=kwargs)