from oslo_utils import encodeutils
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient.v2 import schemas
class ObjectController(object):

    def __init__(self, http_client, schema_client):
        self.http_client = http_client
        self.schema_client = schema_client

    @utils.memoized_property
    def model(self):
        schema = self.schema_client.get('metadefs/object')
        return warlock.model_factory(schema.raw(), base_class=schemas.SchemaBasedModel)

    @utils.add_req_id_to_object()
    def create(self, namespace, **kwargs):
        """Create an object.

        :param namespace: Name of a namespace the object belongs.
        :param kwargs: Unpacked object.
        """
        try:
            obj = self.model(kwargs)
        except (warlock.InvalidOperation, ValueError) as e:
            raise TypeError(encodeutils.exception_to_unicode(e))
        url = '/v2/metadefs/namespaces/%(namespace)s/objects' % {'namespace': namespace}
        resp, body = self.http_client.post(url, data=obj)
        body.pop('self', None)
        return (self.model(**body), resp)

    def update(self, namespace, object_name, **kwargs):
        """Update an object.

        :param namespace: Name of a namespace the object belongs.
        :param object_name: Name of an object (old one).
        :param kwargs: Unpacked object.
        """
        obj = self.get(namespace, object_name)
        for key, value in kwargs.items():
            try:
                setattr(obj, key, value)
            except warlock.InvalidOperation as e:
                raise TypeError(encodeutils.exception_to_unicode(e))
        read_only = ['schema', 'updated_at', 'created_at']
        for elem in read_only:
            if elem in obj:
                del obj[elem]
        url = '/v2/metadefs/namespaces/%(namespace)s/objects/%(object_name)s' % {'namespace': namespace, 'object_name': object_name}
        resp, _ = self.http_client.put(url, data=obj.wrapped)
        req_id_hdr = {'x-openstack-request-id': utils._extract_request_id(resp)}
        return self._get(namespace, obj.name, req_id_hdr)

    def get(self, namespace, object_name):
        return self._get(namespace, object_name)

    @utils.add_req_id_to_object()
    def _get(self, namespace, object_name, header=None):
        url = '/v2/metadefs/namespaces/%(namespace)s/objects/%(object_name)s' % {'namespace': namespace, 'object_name': object_name}
        header = header or {}
        resp, body = self.http_client.get(url, headers=header)
        body.pop('self', None)
        return (self.model(**body), resp)

    @utils.add_req_id_to_generator()
    def list(self, namespace, **kwargs):
        """Retrieve a listing of metadata objects.

        :returns: generator over list of objects
        """
        url = '/v2/metadefs/namespaces/%(namespace)s/objects' % {'namespace': namespace}
        resp, body = self.http_client.get(url)
        for obj in body['objects']:
            yield (self.model(obj), resp)

    @utils.add_req_id_to_object()
    def delete(self, namespace, object_name):
        """Delete an object."""
        url = '/v2/metadefs/namespaces/%(namespace)s/objects/%(object_name)s' % {'namespace': namespace, 'object_name': object_name}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)

    @utils.add_req_id_to_object()
    def delete_all(self, namespace):
        """Delete all objects in a namespace."""
        url = '/v2/metadefs/namespaces/%(namespace)s/objects' % {'namespace': namespace}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)