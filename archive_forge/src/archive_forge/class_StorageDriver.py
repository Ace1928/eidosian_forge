import errno
import hashlib
import os.path  # pylint: disable-msg=W0404
import warnings
from typing import Dict, List, Type, Iterator, Optional
from os.path import join as pjoin
import libcloud.utils.files
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
class StorageDriver(BaseDriver):
    """
    A base StorageDriver to derive from.
    """
    connectionCls = ConnectionUserAndKey
    name = None
    hash_type = 'md5'
    supports_chunked_encoding = False
    strict_mode = False

    def iterate_containers(self):
        """
        Return a iterator of containers for the given account

        :return: A iterator of Container instances.
        :rtype: ``iterator`` of :class:`libcloud.storage.base.Container`
        """
        raise NotImplementedError('iterate_containers not implemented for this driver')

    def list_containers(self):
        """
        Return a list of containers.

        :return: A list of Container instances.
        :rtype: ``list`` of :class:`Container`
        """
        return list(self.iterate_containers())

    def iterate_container_objects(self, container, prefix=None, ex_prefix=None):
        """
        Return a iterator of objects for the given container.

        :param container: Container instance
        :type container: :class:`libcloud.storage.base.Container`

        :param prefix: Filter objects starting with a prefix.
        :type  prefix: ``str``

        :param ex_prefix: (Deprecated.) Filter objects starting with a prefix.
        :type  ex_prefix: ``str``

        :return: A iterator of Object instances.
        :rtype: ``iterator`` of :class:`libcloud.storage.base.Object`
        """
        raise NotImplementedError('iterate_container_objects not implemented for this driver')

    def list_container_objects(self, container, prefix=None, ex_prefix=None):
        """
        Return a list of objects for the given container.

        :param container: Container instance.
        :type container: :class:`libcloud.storage.base.Container`

        :param prefix: Filter objects starting with a prefix.
        :type  prefix: ``str``

        :param ex_prefix: (Deprecated.) Filter objects starting with a prefix.
        :type  ex_prefix: ``str``

        :return: A list of Object instances.
        :rtype: ``list`` of :class:`libcloud.storage.base.Object`
        """
        return list(self.iterate_container_objects(container, prefix=prefix, ex_prefix=ex_prefix))

    def _normalize_prefix_argument(self, prefix, ex_prefix):
        if ex_prefix:
            warnings.warn('The ``ex_prefix`` argument is deprecated - please update code to use ``prefix``', DeprecationWarning)
            return ex_prefix
        return prefix

    def _filter_listed_container_objects(self, objects, prefix):
        if prefix is not None:
            warnings.warn('Driver %s does not implement native object filtering; falling back to filtering the full object stream.' % self.__class__.__name__)
        for obj in objects:
            if prefix is None or obj.name.startswith(prefix):
                yield obj

    def get_container(self, container_name):
        """
        Return a container instance.

        :param container_name: Container name.
        :type container_name: ``str``

        :return: :class:`Container` instance.
        :rtype: :class:`libcloud.storage.base.Container`
        """
        raise NotImplementedError('get_object not implemented for this driver')

    def get_container_cdn_url(self, container):
        """
        Return a container CDN URL.

        :param container: Container instance
        :type  container: :class:`libcloud.storage.base.Container`

        :return: A CDN URL for this container.
        :rtype: ``str``
        """
        raise NotImplementedError('get_container_cdn_url not implemented for this driver')

    def get_object(self, container_name, object_name):
        """
        Return an object instance.

        :param container_name: Container name.
        :type  container_name: ``str``

        :param object_name: Object name.
        :type  object_name: ``str``

        :return: :class:`Object` instance.
        :rtype: :class:`libcloud.storage.base.Object`
        """
        raise NotImplementedError('get_object not implemented for this driver')

    def get_object_cdn_url(self, obj):
        """
        Return an object CDN URL.

        :param obj: Object instance
        :type  obj: :class:`libcloud.storage.base.Object`

        :return: A CDN URL for this object.
        :rtype: ``str``
        """
        raise NotImplementedError('get_object_cdn_url not implemented for this driver')

    def enable_container_cdn(self, container):
        """
        Enable container CDN.

        :param container: Container instance
        :type  container: :class:`libcloud.storage.base.Container`

        :rtype: ``bool``
        """
        raise NotImplementedError('enable_container_cdn not implemented for this driver')

    def enable_object_cdn(self, obj):
        """
        Enable object CDN.

        :param obj: Object instance
        :type  obj: :class:`libcloud.storage.base.Object`

        :rtype: ``bool``
        """
        raise NotImplementedError('enable_object_cdn not implemented for this driver')

    def download_object(self, obj, destination_path, overwrite_existing=False, delete_on_failure=True):
        """
        Download an object to the specified destination path.

        :param obj: Object instance.
        :type obj: :class:`libcloud.storage.base.Object`

        :param destination_path: Full path to a file or a directory where the
                                 incoming file will be saved.
        :type destination_path: ``str``

        :param overwrite_existing: True to overwrite an existing file,
                                   defaults to False.
        :type overwrite_existing: ``bool``

        :param delete_on_failure: True to delete a partially downloaded file if
                                   the download was not successful (hash
                                   mismatch / file size).
        :type delete_on_failure: ``bool``

        :return: True if an object has been successfully downloaded, False
                 otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('download_object not implemented for this driver')

    def download_object_as_stream(self, obj, chunk_size=None):
        """
        Return a iterator which yields object data.

        :param obj: Object instance
        :type obj: :class:`libcloud.storage.base.Object`

        :param chunk_size: Optional chunk size (in bytes).
        :type chunk_size: ``int``

        :rtype: ``iterator`` of ``bytes``
        """
        raise NotImplementedError('download_object_as_stream not implemented for this driver')

    def download_object_range(self, obj, destination_path, start_bytes, end_bytes=None, overwrite_existing=False, delete_on_failure=True):
        """
        Download part of an object.

        :param obj: Object instance.
        :type obj: :class:`libcloud.storage.base.Object`

        :param destination_path: Full path to a file or a directory where the
                                 incoming file will be saved.
        :type destination_path: ``str``

        :param start_bytes: Start byte offset (inclusive) for the range
                            download. Offset is 0 index based so the first
                            byte in file file is "0".
        :type start_bytes: ``int``

        :param end_bytes: End byte offset (non-inclusive) for the range
                          download. If not provided, it will default to the
                          end of the file.
        :type end_bytes: ``int``

        :param overwrite_existing: True to overwrite an existing file,
                                   defaults to False.
        :type overwrite_existing: ``bool``

        :param delete_on_failure: True to delete a partially downloaded file if
                                   the download was not successful (hash
                                   mismatch / file size).
        :type delete_on_failure: ``bool``

        :return: True if an object has been successfully downloaded, False
                 otherwise.
        :rtype: ``bool``

        """
        raise NotImplementedError('download_object_range not implemented for this driver')

    def download_object_range_as_stream(self, obj, start_bytes, end_bytes=None, chunk_size=None):
        """
        Return a iterator which yields range / part of the object data.

        :param obj: Object instance
        :type obj: :class:`libcloud.storage.base.Object`

        :param start_bytes: Start byte offset (inclusive) for the range
                            download. Offset is 0 index based so the first
                            byte in file file is "0".
        :type start_bytes: ``int``

        :param end_bytes: End byte offset (non-inclusive) for the range
                          download. If not provided, it will default to the
                          end of the file.
        :type end_bytes: ``int``

        :param chunk_size: Optional chunk size (in bytes).
        :type chunk_size: ``int``

        :rtype: ``iterator`` of ``bytes``
        """
        raise NotImplementedError('download_object_range_as_stream not implemented for this driver')

    def upload_object(self, file_path, container, object_name, extra=None, verify_hash=True, headers=None):
        """
        Upload an object currently located on a disk.

        :param file_path: Path to the object on disk.
        :type file_path: ``str``

        :param container: Destination container.
        :type container: :class:`libcloud.storage.base.Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :param verify_hash: Verify hash
        :type verify_hash: ``bool``

        :param extra: Extra attributes (driver specific). (optional)
        :type extra: ``dict``

        :param headers: (optional) Additional request headers,
            such as CORS headers. For example:
            headers = {'Access-Control-Allow-Origin': 'http://mozilla.com'}
        :type headers: ``dict``

        :rtype: :class:`libcloud.storage.base.Object`
        """
        raise NotImplementedError('upload_object not implemented for this driver')

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, headers=None):
        """
        Upload an object using an iterator.

        If a provider supports it, chunked transfer encoding is used and you
        don't need to know in advance the amount of data to be uploaded.

        Otherwise if a provider doesn't support it, iterator will be exhausted
        so a total size for data to be uploaded can be determined.

        Note: Exhausting the iterator means that the whole data must be
        buffered in memory which might result in memory exhausting when
        uploading a very large object.

        If a file is located on a disk you are advised to use upload_object
        function which uses fs.stat function to determine the file size and it
        doesn't need to buffer whole object in the memory.

        :param iterator: An object which implements the iterator interface.
        :type iterator: :class:`object`

        :param container: Destination container.
        :type container: :class:`libcloud.storage.base.Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :param extra: (optional) Extra attributes (driver specific). Note:
            This dictionary must contain a 'content_type' key which represents
            a content type of the stored object.
        :type extra: ``dict``

        :param headers: (optional) Additional request headers,
            such as CORS headers. For example:
            headers = {'Access-Control-Allow-Origin': 'http://mozilla.com'}
        :type headers: ``dict``

        :rtype: ``libcloud.storage.base.Object``
        """
        raise NotImplementedError('upload_object_via_stream not implemented for this driver')

    def delete_object(self, obj):
        """
        Delete an object.

        :param obj: Object instance.
        :type obj: :class:`libcloud.storage.base.Object`

        :return: ``bool`` True on success.
        :rtype: ``bool``
        """
        raise NotImplementedError('delete_object not implemented for this driver')

    def create_container(self, container_name):
        """
        Create a new container.

        :param container_name: Container name.
        :type container_name: ``str``

        :return: Container instance on success.
        :rtype: :class:`libcloud.storage.base.Container`
        """
        raise NotImplementedError('create_container not implemented for this driver')

    def delete_container(self, container):
        """
        Delete a container.

        :param container: Container instance
        :type container: :class:`libcloud.storage.base.Container`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('delete_container not implemented for this driver')

    def _get_object(self, obj, callback, callback_kwargs, response, success_status_code=None):
        """
        Call passed callback and start transfer of the object'

        :param obj: Object instance.
        :type obj: :class:`Object`

        :param callback: Function which is called with the passed
            callback_kwargs
        :type callback: :class:`function`

        :param callback_kwargs: Keyword arguments which are passed to the
             callback.
        :type callback_kwargs: ``dict``

        :param response: Response instance.
        :type response: :class:`Response`

        :param success_status_code: Status code which represents a successful
                                    transfer (defaults to httplib.OK)
        :type success_status_code: ``int``

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        success_status_code = success_status_code or httplib.OK
        if not isinstance(success_status_code, (list, tuple)):
            success_status_codes = [success_status_code]
        else:
            success_status_codes = success_status_code
        if response.status in success_status_codes:
            return callback(**callback_kwargs)
        elif response.status == httplib.NOT_FOUND:
            raise ObjectDoesNotExistError(object_name=obj.name, value='', driver=self)
        raise LibcloudError(value='Unexpected status code: %s' % response.status, driver=self)

    def _save_object(self, response, obj, destination_path, overwrite_existing=False, delete_on_failure=True, chunk_size=None, partial_download=False):
        """
        Save object to the provided path.

        :param response: RawResponse instance.
        :type response: :class:`RawResponse`

        :param obj: Object instance.
        :type obj: :class:`Object`

        :param destination_path: Destination directory.
        :type destination_path: ``str``

        :param delete_on_failure: True to delete partially downloaded object if
                                  the download fails.
        :type delete_on_failure: ``bool``

        :param overwrite_existing: True to overwrite a local path if it already
                                   exists.
        :type overwrite_existing: ``bool``

        :param chunk_size: Optional chunk size
            (defaults to ``libcloud.storage.base.CHUNK_SIZE``, 8kb)
        :type chunk_size: ``int``

        :param partial_download: True if this is a range (partial) save,
                                 False otherwise.
        :type partial_download: ``bool``

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        chunk_size = chunk_size or CHUNK_SIZE
        base_name = os.path.basename(destination_path)
        if not base_name and (not os.path.exists(destination_path)):
            raise LibcloudError(value='Path %s does not exist' % destination_path, driver=self)
        if not base_name:
            file_path = pjoin(destination_path, obj.name)
        else:
            file_path = destination_path
        if os.path.exists(file_path) and (not overwrite_existing):
            raise LibcloudError(value='File %s already exists, but ' % file_path + 'overwrite_existing=False', driver=self)
        bytes_transferred = 0
        with open(file_path, 'wb') as file_handle:
            for chunk in response._response.iter_content(chunk_size):
                file_handle.write(b(chunk))
                bytes_transferred += len(chunk)
        if not partial_download and int(obj.size) != int(bytes_transferred):
            if delete_on_failure:
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
            return False
        return True

    def _upload_object(self, object_name, content_type, request_path, request_method='PUT', headers=None, file_path=None, stream=None, chunked=False, multipart=False):
        """
        Helper function for setting common request headers and calling the
        passed in callback which uploads an object.
        """
        headers = headers or {}
        if file_path and (not os.path.exists(file_path)):
            raise OSError('File %s does not exist' % file_path)
        if stream is not None and (not hasattr(stream, 'next')) and (not hasattr(stream, '__next__')):
            raise AttributeError('iterator object must implement next() ' + 'method.')
        headers['Content-Type'] = self._determine_content_type(content_type, object_name, file_path=file_path)
        if stream:
            response = self.connection.request(request_path, method=request_method, data=stream, headers=headers, raw=True)
            stream_hash, stream_length = self._hash_buffered_stream(stream, self._get_hash_function())
        else:
            with open(file_path, 'rb') as file_stream:
                response = self.connection.request(request_path, method=request_method, data=file_stream, headers=headers, raw=True)
            with open(file_path, 'rb') as file_stream:
                stream_hash, stream_length = self._hash_buffered_stream(file_stream, self._get_hash_function())
        return {'response': response, 'bytes_transferred': stream_length, 'data_hash': stream_hash}

    def _determine_content_type(self, content_type, object_name, file_path=None):
        if content_type:
            return content_type
        name = file_path or object_name
        content_type, _ = libcloud.utils.files.guess_file_mime_type(name)
        if self.strict_mode and (not content_type):
            raise AttributeError('File content-type could not be guessed for "%s" and no content_type value is provided' % name)
        return content_type or DEFAULT_CONTENT_TYPE

    def _hash_buffered_stream(self, stream, hasher, blocksize=65536):
        total_len = 0
        if hasattr(stream, '__next__') or hasattr(stream, 'next'):
            if hasattr(stream, 'seek'):
                try:
                    stream.seek(0)
                except OSError as e:
                    if e.errno != errno.ESPIPE:
                        raise e
            for chunk in libcloud.utils.files.read_in_chunks(iterator=stream):
                hasher.update(b(chunk))
                total_len += len(chunk)
            return (hasher.hexdigest(), total_len)
        if not hasattr(stream, '__exit__'):
            for s in stream:
                hasher.update(s)
                total_len = total_len + len(s)
            return (hasher.hexdigest(), total_len)
        with stream:
            buf = stream.read(blocksize)
            while len(buf) > 0:
                total_len = total_len + len(buf)
                hasher.update(buf)
                buf = stream.read(blocksize)
        return (hasher.hexdigest(), total_len)

    def _get_hash_function(self):
        """
        Return instantiated hash function for the hash type supported by
        the provider.
        """
        try:
            func = getattr(hashlib, self.hash_type)()
        except AttributeError:
            raise RuntimeError('Invalid or unsupported hash type: %s' % self.hash_type)
        return func

    def _validate_start_and_end_bytes(self, start_bytes, end_bytes=None):
        """
        Method which validates that start_bytes and end_bytes arguments contain
        valid values.
        """
        if start_bytes < 0:
            raise ValueError('start_bytes must be greater than 0')
        if end_bytes is not None:
            if start_bytes > end_bytes:
                raise ValueError('start_bytes must be smaller than end_bytes')
            elif start_bytes == end_bytes:
                raise ValueError("start_bytes and end_bytes can't be the same. end_bytes is non-inclusive")
        return True

    def _get_standard_range_str(self, start_bytes, end_bytes=None, end_bytes_inclusive=False):
        """
        Return range string which is used as a Range header value for range
        requests for drivers which follow standard Range header notation

        This returns range string in the following format:
        bytes=<start_bytes>-<end bytes>.

        For example:

        bytes=1-10
        bytes=0-2
        bytes=5-
        bytes=100-5000

        :param end_bytes_inclusive: True if "end_bytes" offset should be
        inclusive (aka opposite from the Python indexing behavior where the end
        index is not inclusive).
        """
        range_str = 'bytes=%s-' % start_bytes
        if end_bytes is not None:
            if end_bytes_inclusive:
                range_str += str(end_bytes)
            else:
                range_str += str(end_bytes - 1)
        return range_str