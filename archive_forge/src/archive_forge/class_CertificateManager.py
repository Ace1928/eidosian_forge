from magnumclient.common import base
from magnumclient import exceptions
class CertificateManager(base.Manager):
    resource_class = Certificate

    @staticmethod
    def _path(id=None):
        return '/v1/certificates/%s' % id if id else '/v1/certificates'

    def get(self, cluster_uuid):
        try:
            return self._list(self._path(cluster_uuid))[0]
        except IndexError:
            return None

    def create(self, **kwargs):
        new = {}
        for key, value in kwargs.items():
            if key in CREATION_ATTRIBUTES:
                new[key] = value
            else:
                raise exceptions.InvalidAttribute('Key must be in %s' % ','.join(CREATION_ATTRIBUTES))
        return self._create(self._path(), new)

    def rotate_ca(self, **kwargs):
        return self._update(self._path(id=kwargs['cluster_uuid']))