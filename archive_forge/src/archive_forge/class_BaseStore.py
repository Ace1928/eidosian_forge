import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
class BaseStore(driver.Store):
    _CAPABILITIES = capabilities.BitMasks.RW_ACCESS
    CHUNKSIZE = 65536
    OPTIONS = _SWIFT_OPTS + sutils.swift_opts

    def get_schemes(self):
        return ('swift+https', 'swift', 'swift+http', 'swift+config')

    def configure(self, re_raise_bsc=False):
        if self.backend_group:
            glance_conf = getattr(self.conf, self.backend_group)
        else:
            glance_conf = self.conf.glance_store
        _obj_size = self._option_get('swift_store_large_object_size')
        self.large_object_size = _obj_size * ONE_MB
        _chunk_size = self._option_get('swift_store_large_object_chunk_size')
        self.large_object_chunk_size = _chunk_size * ONE_MB
        self.admin_tenants = glance_conf.swift_store_admin_tenants
        self.region = glance_conf.swift_store_region
        self.service_type = glance_conf.swift_store_service_type
        self.conf_endpoint = glance_conf.swift_store_endpoint
        self.endpoint_type = glance_conf.swift_store_endpoint_type
        self.insecure = glance_conf.swift_store_auth_insecure
        self.ssl_compression = glance_conf.swift_store_ssl_compression
        self.cacert = glance_conf.swift_store_cacert
        if self.insecure:
            self.ks_verify = False
        else:
            self.ks_verify = self.cacert or True
        if swiftclient is None:
            msg = _('Missing dependency python_swiftclient.')
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=msg)
        if glance_conf.swift_buffer_on_upload:
            buffer_dir = glance_conf.swift_upload_buffer_dir
            if buffered.validate_buffering(buffer_dir):
                self.reader_class = buffered.BufferedReader
        else:
            self.reader_class = ChunkReader
        super(BaseStore, self).configure(re_raise_bsc=re_raise_bsc)

    def _get_object(self, location, manager, start=None):
        headers = {}
        if start is not None:
            bytes_range = 'bytes=%d-' % start
            headers = {'Range': bytes_range}
        try:
            resp_headers, resp_body = manager.get_connection().get_object(location.container, location.obj, resp_chunk_size=self.CHUNKSIZE, headers=headers)
        except swiftclient.ClientException as e:
            if e.http_status == http.client.NOT_FOUND:
                msg = _('Swift could not find object %s.') % location.obj
                LOG.warning(msg)
                raise exceptions.NotFound(message=msg)
            else:
                raise
        return (resp_headers, resp_body)

    @capabilities.check
    def get(self, location, connection=None, offset=0, chunk_size=None, context=None):
        if self.backend_group:
            glance_conf = getattr(self.conf, self.backend_group)
        else:
            glance_conf = self.conf.glance_store
        location = location.store_location
        allow_retry = glance_conf.swift_store_retry_get_count > 0
        with self.get_manager(location, context, allow_reauth=allow_retry) as manager:
            resp_headers, resp_body = self._get_object(location, manager=manager)

            class ResponseIndexable(glance_store.Indexable):

                def another(self):
                    try:
                        return next(self.wrapped)
                    except StopIteration:
                        return b''
            length = int(resp_headers.get('content-length', 0))
            if allow_retry:
                resp_body = swift_retry_iter(resp_body, length, self, location, manager=manager)
            return (ResponseIndexable(resp_body, length), length)

    def get_size(self, location, connection=None, context=None):
        location = location.store_location
        if not connection:
            connection = self.get_connection(location, context=context)
        try:
            resp_headers = connection.head_object(location.container, location.obj)
            return int(resp_headers.get('content-length', 0))
        except Exception:
            return 0

    def _option_get(self, param):
        if self.backend_group:
            result = getattr(getattr(self.conf, self.backend_group), param)
        else:
            result = getattr(self.conf.glance_store, param)
        if result is None:
            reason = _('Could not find %(param)s in configuration options.') % param
            LOG.error(reason)
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        return result

    def _delete_stale_chunks(self, connection, container, chunk_list):
        for chunk in chunk_list:
            LOG.debug('Deleting chunk %s' % chunk)
            try:
                connection.delete_object(container, chunk)
            except Exception:
                msg = _('Failed to delete orphaned chunk %(container)s/%(chunk)s')
                LOG.exception(msg % {'container': container, 'chunk': chunk})

    @driver.back_compat_add
    @capabilities.check
    def add(self, image_id, image_file, image_size, hashing_algo, context=None, verifier=None):
        """
        Stores an image file with supplied identifier to the backend
        storage system and returns a tuple containing information
        about the stored image.

        :param image_id: The opaque image identifier
        :param image_file: The image data to write, as a file-like object
        :param image_size: The size of the image data to write, in bytes
        :param hashing_algo: A hashlib algorithm identifier (string)
        :param verifier: An object used to verify signatures for images

        :returns: tuple of URL in backing store, bytes written, checksum,
                multihash value, and a dictionary with storage system
                specific information
        :raises: `glance_store.exceptions.Duplicate` if something already
                exists at this location
        """
        os_hash_value = gutils.get_hasher(hashing_algo, False)
        location = self.create_location(image_id, context=context)
        need_chunks = image_size == 0 or image_size >= self.large_object_size
        with self.get_manager(location, context, allow_reauth=need_chunks) as manager:
            self._create_container_if_missing(location.container, manager.get_connection())
            LOG.debug("Adding image object '%(obj_name)s' to Swift" % dict(obj_name=location.obj))
            try:
                if not need_chunks:
                    checksum = gutils.get_hasher('md5', False)
                    reader = ChunkReader(image_file, checksum, os_hash_value, image_size, verifier=verifier)
                    obj_etag = manager.get_connection().put_object(location.container, location.obj, reader, content_length=image_size)
                else:
                    chunk_id = 1
                    if image_size > 0:
                        total_chunks = str(int(math.ceil(float(image_size) / float(self.large_object_chunk_size))))
                    else:
                        LOG.debug('Cannot determine image size because it is either not provided in the request or chunked-transfer encoding is used. Adding image as a segmented object to Swift.')
                        total_chunks = '?'
                    checksum = gutils.get_hasher('md5', False)
                    written_chunks = []
                    combined_chunks_size = 0
                    while True:
                        chunk_size = self.large_object_chunk_size
                        if image_size == 0:
                            content_length = None
                        else:
                            left = image_size - combined_chunks_size
                            if left == 0:
                                break
                            if chunk_size > left:
                                chunk_size = left
                            content_length = chunk_size
                        chunk_name = '%s-%05d' % (location.obj, chunk_id)
                        with self.reader_class(image_file, checksum, os_hash_value, chunk_size, verifier, backend_group=self.backend_group) as reader:
                            if reader.is_zero_size is True:
                                LOG.debug('Not writing zero-length chunk.')
                                break
                            try:
                                chunk_etag = manager.get_connection().put_object(location.container, chunk_name, reader, content_length=content_length)
                                written_chunks.append(chunk_name)
                            except Exception:
                                with excutils.save_and_reraise_exception():
                                    LOG.error(_('Error during chunked upload to backend, deleting stale chunks.'))
                                    self._delete_stale_chunks(manager.get_connection(), location.container, written_chunks)
                            bytes_read = reader.bytes_read
                            msg = 'Wrote chunk %(chunk_name)s (%(chunk_id)d/%(total_chunks)s) of length %(bytes_read)d to Swift returning MD5 of content: %(chunk_etag)s' % {'chunk_name': chunk_name, 'chunk_id': chunk_id, 'total_chunks': total_chunks, 'bytes_read': bytes_read, 'chunk_etag': chunk_etag}
                            LOG.debug(msg)
                        chunk_id += 1
                        combined_chunks_size += bytes_read
                    if image_size == 0:
                        image_size = combined_chunks_size
                    manifest = '%s/%s-' % (location.container, location.obj)
                    headers = {'X-Object-Manifest': manifest}
                    manager.get_connection().put_object(location.container, location.obj, None, headers=headers)
                    obj_etag = checksum.hexdigest()
                if sutils.is_multiple_swift_store_accounts_enabled(self.conf, backend=self.backend_group):
                    include_creds = False
                else:
                    include_creds = True
                metadata = {}
                if self.backend_group:
                    metadata['store'] = self.backend_group
                return (location.get_uri(credentials_included=include_creds), image_size, obj_etag, os_hash_value.hexdigest(), metadata)
            except swiftclient.ClientException as e:
                if e.http_status == http.client.CONFLICT:
                    msg = _('Swift already has an image at this location')
                    raise exceptions.Duplicate(message=msg)
                elif e.http_status == http.client.REQUEST_ENTITY_TOO_LARGE:
                    raise exceptions.StorageFull(message=e.msg)
                msg = _('Failed to add object to Swift.\nGot error from Swift: %s.') % encodeutils.exception_to_unicode(e)
                LOG.error(msg)
                raise glance_store.BackendException(msg)

    @capabilities.check
    def delete(self, location, connection=None, context=None):
        location = location.store_location
        if not connection:
            connection = self.get_connection(location, context=context)
        try:
            dlo_manifest = None
            slo_manifest = None
            try:
                headers = connection.head_object(location.container, location.obj)
                dlo_manifest = headers.get('x-object-manifest')
                slo_manifest = headers.get('x-static-large-object')
            except swiftclient.ClientException as e:
                if e.http_status != http.client.NOT_FOUND:
                    raise
            if _is_slo(slo_manifest):
                query_string = 'multipart-manifest=delete'
                connection.delete_object(location.container, location.obj, query_string=query_string)
                return
            if dlo_manifest:
                obj_container, obj_prefix = dlo_manifest.split('/', 1)
                segments = connection.get_container(obj_container, prefix=obj_prefix)[1]
                for segment in segments:
                    try:
                        connection.delete_object(obj_container, segment['name'])
                    except swiftclient.ClientException:
                        msg = _('Unable to delete segment %(segment_name)s')
                        msg = msg % {'segment_name': segment['name']}
                        LOG.exception(msg)
            connection.delete_object(location.container, location.obj)
        except swiftclient.ClientException as e:
            if e.http_status == http.client.NOT_FOUND:
                msg = _('Swift could not find image at URI.')
                raise exceptions.NotFound(message=msg)
            else:
                raise

    def _create_container_if_missing(self, container, connection):
        """
        Creates a missing container in Swift if the
        ``swift_store_create_container_on_put`` option is set.

        :param container: Name of container to create
        :param connection: Connection to swift service
        """
        if self.backend_group:
            store_conf = getattr(self.conf, self.backend_group)
        else:
            store_conf = self.conf.glance_store
        try:
            connection.head_container(container)
        except swiftclient.ClientException as e:
            if e.http_status == http.client.NOT_FOUND:
                if store_conf.swift_store_create_container_on_put:
                    try:
                        msg = _LI('Creating swift container %(container)s') % {'container': container}
                        LOG.info(msg)
                        connection.put_container(container)
                    except swiftclient.ClientException as e:
                        msg = _('Failed to add container to Swift.\nGot error from Swift: %s.') % encodeutils.exception_to_unicode(e)
                        raise glance_store.BackendException(msg)
                else:
                    msg = _('The container %(container)s does not exist in Swift. Please set the swift_store_create_container_on_put option to add container to Swift automatically.') % {'container': container}
                    raise glance_store.BackendException(msg)
            else:
                raise

    def get_connection(self, location, context=None):
        raise NotImplementedError()

    def create_location(self, image_id, context=None):
        raise NotImplementedError()

    def init_client(self, location, context=None):
        """Initialize and return client to authorize against keystone

        The method invariant is the following: it always returns Keystone
        client that can be used to receive fresh token in any time. Otherwise
        it raises appropriate exception.
        :param location: swift location data
        :param context: user context (it is not required if user grants are
        specified for single tenant store)
        :return correctly initialized keystone client
        """
        raise NotImplementedError()

    def get_store_connection(self, auth_token, storage_url):
        """Get initialized swift connection

        :param auth_token: auth token
        :param storage_url: swift storage url
        :return: swiftclient connection that allows to request container and
                 others
        """
        return swiftclient.Connection(preauthurl=storage_url, preauthtoken=auth_token, insecure=self.insecure, ssl_compression=self.ssl_compression, cacert=self.cacert)

    def get_manager(self, store_location, context=None, allow_reauth=False):
        """Return appropriate connection manager for store

        The method detects store type (singletenant or multitenant) and returns
        appropriate connection manager (singletenant or multitenant) that
        allows to request swiftclient connections.

        :param store_location: StoreLocation object that define image location
        :param context: user context
        :param allow_reauth: defines if we allow re-authentication when user
            token is expired and refresh swift connection

        :return: connection manager for store
        """
        msg = _('There is no Connection Manager implemented for %s class.')
        raise NotImplementedError(msg % self.__class__.__name__)

    def _set_url_prefix(self, context=None):
        raise NotImplementedError()