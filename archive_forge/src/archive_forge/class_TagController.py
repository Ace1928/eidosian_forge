from oslo_utils import encodeutils
import urllib.parse
import warlock
from glanceclient.common import utils
from glanceclient.v2 import schemas
class TagController(object):

    def __init__(self, http_client, schema_client):
        self.http_client = http_client
        self.schema_client = schema_client

    @utils.memoized_property
    def model(self):
        schema = self.schema_client.get('metadefs/tag')
        return warlock.model_factory(schema.raw(), base_class=schemas.SchemaBasedModel)

    @utils.add_req_id_to_object()
    def create(self, namespace, tag_name):
        """Create a tag.

        :param namespace: Name of a namespace the Tag belongs.
        :param tag_name: The name of the new tag to create.
        """
        url = '/v2/metadefs/namespaces/%(namespace)s/tags/%(tag_name)s' % {'namespace': namespace, 'tag_name': tag_name}
        resp, body = self.http_client.post(url)
        body.pop('self', None)
        return (self.model(**body), resp)

    @utils.add_req_id_to_generator()
    def create_multiple(self, namespace, **kwargs):
        """Create the list of tags.

        :param namespace: Name of a namespace to which the Tags belong.
        :param kwargs: list of tags, optional parameter append.
        """
        tag_names = kwargs.pop('tags', [])
        md_tag_list = []
        for tag_name in tag_names:
            try:
                md_tag_list.append(self.model(name=tag_name))
            except warlock.InvalidOperation as e:
                raise TypeError(encodeutils.exception_to_unicode(e))
        tags = {'tags': md_tag_list}
        headers = {}
        url = '/v2/metadefs/namespaces/%(namespace)s/tags' % {'namespace': namespace}
        append = kwargs.pop('append', False)
        if append:
            headers['X-Openstack-Append'] = True
        resp, body = self.http_client.post(url, headers=headers, data=tags)
        body.pop('self', None)
        for tag in body['tags']:
            yield (self.model(tag), resp)

    def update(self, namespace, tag_name, **kwargs):
        """Update a tag.

        :param namespace: Name of a namespace the Tag belongs.
        :param tag_name: Name of the Tag (old one).
        :param kwargs: Unpacked tag.
        """
        tag = self.get(namespace, tag_name)
        for key, value in kwargs.items():
            try:
                setattr(tag, key, value)
            except warlock.InvalidOperation as e:
                raise TypeError(encodeutils.exception_to_unicode(e))
        read_only = ['updated_at', 'created_at']
        for elem in read_only:
            if elem in tag:
                del tag[elem]
        url = '/v2/metadefs/namespaces/%(namespace)s/tags/%(tag_name)s' % {'namespace': namespace, 'tag_name': tag_name}
        resp, _ = self.http_client.put(url, data=tag.wrapped)
        req_id_hdr = {'x-openstack-request-id': utils._extract_request_id(resp)}
        return self._get(namespace, tag.name, req_id_hdr)

    def get(self, namespace, tag_name):
        return self._get(namespace, tag_name)

    @utils.add_req_id_to_object()
    def _get(self, namespace, tag_name, header=None):
        url = '/v2/metadefs/namespaces/%(namespace)s/tags/%(tag_name)s' % {'namespace': namespace, 'tag_name': tag_name}
        header = header or {}
        resp, body = self.http_client.get(url, headers=header)
        body.pop('self', None)
        return (self.model(**body), resp)

    @utils.add_req_id_to_generator()
    def list(self, namespace, **kwargs):
        """Retrieve a listing of metadata tags.

        :returns: generator over list of tags.
        """
        url = '/v2/metadefs/namespaces/%(namespace)s/tags' % {'namespace': namespace}
        resp, body = self.http_client.get(url)
        for tag in body['tags']:
            yield (self.model(tag), resp)

    @utils.add_req_id_to_object()
    def delete(self, namespace, tag_name):
        """Delete a tag."""
        url = '/v2/metadefs/namespaces/%(namespace)s/tags/%(tag_name)s' % {'namespace': namespace, 'tag_name': tag_name}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)

    @utils.add_req_id_to_object()
    def delete_all(self, namespace):
        """Delete all tags in a namespace."""
        url = '/v2/metadefs/namespaces/%(namespace)s/tags' % {'namespace': namespace}
        resp, body = self.http_client.delete(url)
        return ((resp, body), resp)