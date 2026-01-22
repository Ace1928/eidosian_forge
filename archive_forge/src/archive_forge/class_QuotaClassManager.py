from zunclient.common import base
class QuotaClassManager(base.Manager):
    resource_class = QuotaClass

    @staticmethod
    def _path(quota_class_name):
        return '/v1/quota_classes/{}'.format(quota_class_name)

    def get(self, quota_class_name):
        return self._list(self._path(quota_class_name))[0]

    def update(self, quota_class_name, containers=None, memory=None, cpu=None, disk=None):
        resources = {}
        if cpu is not None:
            resources['cpu'] = cpu
        if memory is not None:
            resources['memory'] = memory
        if containers is not None:
            resources['containers'] = containers
        if disk is not None:
            resources['disk'] = disk
        return self._update(self._path(quota_class_name), resources, method='PUT')