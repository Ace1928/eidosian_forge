from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
class FromRegistry(DockerImage):
    """This accesses a docker image hosted on a registry (non-local)."""

    def __init__(self, name, basic_creds, transport, accepted_mimes=docker_http.MANIFEST_SCHEMA2_MIMES):
        super().__init__()
        self._name = name
        self._creds = basic_creds
        self._original_transport = transport
        self._accepted_mimes = accepted_mimes
        self._response = {}

    def _content(self, suffix, accepted_mimes=None, cache=True):
        """Fetches content of the resources from registry by http calls."""
        if isinstance(self._name, docker_name.Repository):
            suffix = '{repository}/{suffix}'.format(repository=self._name.repository, suffix=suffix)
        if suffix in self._response:
            return self._response[suffix]
        _, content = self._transport.Request('{scheme}://{registry}/v2/{suffix}'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry, suffix=suffix), accepted_codes=[six.moves.http_client.OK], accepted_mimes=accepted_mimes)
        if cache:
            self._response[suffix] = content
        return content

    def check_usage_only(self):
        response = json.loads(self._content('tags/list?check_usage_only=true').decode('utf8'))
        if 'usage' not in response:
            raise docker_http.BadStateException('Malformed JSON response: {}. Missing "usage" field'.format(response))
        return response.get('usage')

    def _tags(self):
        return json.loads(self._content('tags/list').decode('utf8'))

    def tags(self):
        return self._tags().get('tags', [])

    def manifests(self):
        payload = self._tags()
        if 'manifest' not in payload:
            return {}
        return payload['manifest']

    def children(self):
        payload = self._tags()
        if 'child' not in payload:
            return []
        return payload['child']

    def exists(self):
        try:
            manifest = json.loads(self.manifest(validate=False))
            return manifest['schemaVersion'] == 2 and 'layers' in manifest and (self.media_type() in self._accepted_mimes)
        except docker_http.V2DiagnosticException as err:
            if err.status == six.moves.http_client.NOT_FOUND:
                return False
            raise

    def digest(self):
        """The digest of the manifest."""
        if isinstance(self._name, docker_name.Digest):
            return self._name.digest
        return super().digest()

    def manifest(self, validate=True):
        """Override."""
        if isinstance(self._name, docker_name.Tag):
            path = 'manifests/' + self._name.tag
            return self._content(path, self._accepted_mimes).decode('utf8')
        else:
            assert isinstance(self._name, docker_name.Digest)
            c = self._content('manifests/' + self._name.digest, self._accepted_mimes)
            computed = docker_digest.SHA256(c)
            if validate and computed != self._name.digest:
                raise DigestMismatchedError("The returned manifest's digest did not match requested digest, %s vs. %s" % (self._name.digest, computed))
            return c.decode('utf8')

    def config_file(self):
        """Override."""
        return self.blob(self.config_blob()).decode('utf8')

    def blob_size(self, digest):
        """The byte size of the raw blob."""
        suffix = 'blobs/' + digest
        if isinstance(self._name, docker_name.Repository):
            suffix = '{repository}/{suffix}'.format(repository=self._name.repository, suffix=suffix)
        resp, unused_content = self._transport.Request('{scheme}://{registry}/v2/{suffix}'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry, suffix=suffix), method='HEAD', accepted_codes=[six.moves.http_client.OK])
        return int(resp['content-length'])

    def blob(self, digest):
        """Override."""
        c = self._content('blobs/' + digest, cache=False)
        computed = docker_digest.SHA256(c)
        if digest != computed:
            raise DigestMismatchedError("The returned content's digest did not match its content-address, %s vs. %s" % (digest, computed if c else '(content was empty)'))
        return c

    def catalog(self, page_size=100):
        if isinstance(self._name, docker_name.Repository):
            raise ValueError('Expected docker_name.Registry for "name"')
        url = '{scheme}://{registry}/v2/_catalog?n={page_size}'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry, page_size=page_size)
        for _, content in self._transport.PaginatedRequest(url, accepted_codes=[six.moves.http_client.OK]):
            wrapper_object = json.loads(content.decode('utf8'))
            if 'repositories' not in wrapper_object:
                raise docker_http.BadStateException('Malformed JSON response: %s' % content)
            for repo in wrapper_object['repositories']:
                yield repo

    def __enter__(self):
        self._transport = docker_http.Transport(self._name, self._creds, self._original_transport, docker_http.PULL)
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass

    def __str__(self):
        return '<docker_image.FromRegistry name: {}>'.format(str(self._name))