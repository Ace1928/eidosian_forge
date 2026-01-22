from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import concurrent.futures
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_image_list as image_list
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
class Push(object):
    """Push encapsulates a Registry v2.2 Docker push session."""

    def __init__(self, name, creds, transport, mount=None, threads=1):
        """Constructor.

    If multiple threads are used, the caller *must* ensure that the provided
    transport is thread-safe, as well as the image that is being uploaded.
    It is notable that tarfile and httplib2.Http in Python are NOT threadsafe.

    Args:
      name: the fully-qualified name of the tag to push
      creds: credential provider for authorizing requests
      transport: the http transport to use for sending requests
      mount: list of repos from which to mount blobs.
      threads: the number of threads to use for uploads.

    Raises:
      ValueError: an incorrectly typed argument was supplied.
    """
        self._name = name
        self._transport = docker_http.Transport(name, creds, transport, docker_http.PUSH)
        self._mount = mount
        self._threads = threads

    def name(self):
        return self._name

    def _scheme_and_host(self):
        return '{scheme}://{registry}'.format(scheme=docker_http.Scheme(self._name.registry), registry=self._name.registry)

    def _base_url(self):
        return self._scheme_and_host() + '/v2/{repository}'.format(repository=self._name.repository)

    def _get_absolute_url(self, location):
        return six.moves.urllib.parse.urljoin(base=self._scheme_and_host(), url=location)

    def blob_exists(self, digest):
        """Check the remote for the given layer."""
        resp, unused_content = self._transport.Request('{base_url}/blobs/{digest}'.format(base_url=self._base_url(), digest=digest), method='HEAD', accepted_codes=[six.moves.http_client.OK, six.moves.http_client.NOT_FOUND])
        return resp.status == six.moves.http_client.OK

    def manifest_exists(self, image):
        """Check the remote for the given manifest by digest."""
        resp, unused_content = self._transport.Request('{base_url}/manifests/{digest}'.format(base_url=self._base_url(), digest=image.digest()), method='GET', accepted_codes=[six.moves.http_client.OK, six.moves.http_client.NOT_FOUND], accepted_mimes=[image.media_type()])
        return resp.status == six.moves.http_client.OK

    def _get_blob(self, image, digest):
        if digest == image.config_blob():
            return image.config_file().encode('utf8')
        return image.blob(digest)

    def _monolithic_upload(self, image, digest):
        self._transport.Request('{base_url}/blobs/uploads/?digest={digest}'.format(base_url=self._base_url(), digest=digest), method='POST', body=self._get_blob(image, digest), accepted_codes=[six.moves.http_client.CREATED])

    def _add_digest(self, url, digest):
        scheme, netloc, path, query_string, fragment = six.moves.urllib.parse.urlsplit(url)
        qs = six.moves.urllib.parse.parse_qs(query_string)
        qs['digest'] = [digest]
        query_string = six.moves.urllib.parse.urlencode(qs, doseq=True)
        return six.moves.urllib.parse.urlunsplit((scheme, netloc, path, query_string, fragment))

    def _put_upload(self, image, digest):
        mounted, location = self._start_upload(digest, self._mount)
        if mounted:
            logging.info('Layer %s mounted.', digest)
            return
        location = self._add_digest(location, digest)
        self._transport.Request(location, method='PUT', body=self._get_blob(image, digest), accepted_codes=[six.moves.http_client.CREATED])

    def patch_upload(self, source, digest):
        mounted, location = self._start_upload(digest, self._mount)
        if mounted:
            logging.info('Layer %s mounted.', digest)
            return
        location = self._get_absolute_url(location)
        blob = source
        if isinstance(source, docker_image.DockerImage):
            blob = self._get_blob(source, digest)
        resp, unused_content = self._transport.Request(location, method='PATCH', body=blob, content_type='application/octet-stream', accepted_codes=[six.moves.http_client.NO_CONTENT, six.moves.http_client.ACCEPTED, six.moves.http_client.CREATED])
        location = self._add_digest(resp['location'], digest)
        location = self._get_absolute_url(location)
        self._transport.Request(location, method='PUT', body=None, accepted_codes=[six.moves.http_client.CREATED])

    def _put_blob(self, image, digest):
        """Upload the aufs .tgz for a single layer."""
        self.patch_upload(image, digest)

    def _remote_tag_digest(self, image):
        """Check the remote for the given manifest by digest."""
        resp, unused_content = self._transport.Request('{base_url}/manifests/{tag}'.format(base_url=self._base_url(), tag=self._name.tag), method='GET', accepted_codes=[six.moves.http_client.OK, six.moves.http_client.NOT_FOUND], accepted_mimes=[image.media_type()])
        if resp.status == six.moves.http_client.NOT_FOUND:
            return None
        return resp.get('docker-content-digest')

    def put_manifest(self, image, use_digest=False):
        """Upload the manifest for this image."""
        if use_digest:
            tag_or_digest = image.digest()
        else:
            tag_or_digest = _tag_or_digest(self._name)
        self._transport.Request('{base_url}/manifests/{tag_or_digest}'.format(base_url=self._base_url(), tag_or_digest=tag_or_digest), method='PUT', body=image.manifest(), content_type=image.media_type(), accepted_codes=[six.moves.http_client.OK, six.moves.http_client.CREATED, six.moves.http_client.ACCEPTED])

    def _start_upload(self, digest, mount=None):
        """POST to begin the upload process with optional cross-repo mount param."""
        if not mount:
            url = '{base_url}/blobs/uploads/'.format(base_url=self._base_url())
            accepted_codes = [six.moves.http_client.ACCEPTED]
        else:
            mount_from = '&'.join(['from=' + six.moves.urllib.parse.quote(repo.repository, '') for repo in self._mount])
            url = '{base_url}/blobs/uploads/?mount={digest}&{mount_from}'.format(base_url=self._base_url(), digest=digest, mount_from=mount_from)
            accepted_codes = [six.moves.http_client.CREATED, six.moves.http_client.ACCEPTED]
        resp, unused_content = self._transport.Request(url, method='POST', body=None, accepted_codes=accepted_codes)
        return (resp.status == six.moves.http_client.CREATED, resp.get('location'))

    def _upload_one(self, image, digest):
        """Upload a single layer, after checking whether it exists already."""
        if self.blob_exists(digest):
            logging.info('Layer %s exists, skipping', digest)
            return
        self._put_blob(image, digest)
        logging.info('Layer %s pushed.', digest)

    def upload(self, image, use_digest=False):
        """Upload the layers of the given image.

    Args:
      image: the image to upload.
      use_digest: use the manifest digest (i.e. not tag) as the image reference.
    """
        if self.manifest_exists(image):
            if isinstance(self._name, docker_name.Tag):
                if self._remote_tag_digest(image) == image.digest():
                    logging.info('Tag points to the right manifest, skipping push.')
                    return
                logging.info('Manifest exists, skipping blob uploads and pushing tag.')
            else:
                logging.info('Manifest exists, skipping upload.')
        elif isinstance(image, image_list.DockerImageList):
            for _, child in image:
                with child:
                    self.upload(child, use_digest=True)
        elif self._threads == 1:
            for digest in image.distributable_blob_set():
                self._upload_one(image, digest)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._threads) as executor:
                future_to_params = {executor.submit(self._upload_one, image, digest): (image, digest) for digest in image.distributable_blob_set()}
                for future in concurrent.futures.as_completed(future_to_params):
                    future.result()
        self.put_manifest(image, use_digest=use_digest)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, unused_value, unused_traceback):
        if exception_type:
            logging.error('Error during upload of: %s', self._name)
            return
        logging.info('Finished upload of: %s', self._name)