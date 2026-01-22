import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
class ExternalStorageRayStorageImpl(ExternalStorage):
    """Implements the external storage interface using the ray storage API."""

    def __init__(self, session_name: str, buffer_size=1024 * 1024, _force_storage_for_testing: Optional[str]=None):
        from ray._private import storage
        if _force_storage_for_testing:
            storage._reset()
            storage._init_storage(_force_storage_for_testing, True)
        self._fs, storage_prefix = storage._get_filesystem_internal()
        self._buffer_size = buffer_size
        self._prefix = os.path.join(storage_prefix, 'spilled_objects', session_name)
        self._fs.create_dir(self._prefix)

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        if len(object_refs) == 0:
            return []
        filename = _get_unique_spill_filename(object_refs)
        url = f'{os.path.join(self._prefix, filename)}'
        with self._fs.open_output_stream(url, buffer_size=self._buffer_size) as f:
            return self._write_multiple_objects(f, object_refs, owner_addresses, url)

    def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
        total = 0
        for i in range(len(object_refs)):
            object_ref = object_refs[i]
            url_with_offset = url_with_offset_list[i].decode()
            parsed_result = parse_url_with_offset(url_with_offset)
            base_url = parsed_result.base_url
            offset = parsed_result.offset
            with self._fs.open_input_file(base_url) as f:
                f.seek(offset)
                address_len = int.from_bytes(f.read(8), byteorder='little')
                metadata_len = int.from_bytes(f.read(8), byteorder='little')
                buf_len = int.from_bytes(f.read(8), byteorder='little')
                self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
                total += buf_len
                owner_address = f.read(address_len)
                metadata = f.read(metadata_len)
                self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
        return total

    def delete_spilled_objects(self, urls: List[str]):
        for url in urls:
            path = parse_url_with_offset(url.decode()).base_url
            try:
                self._fs.delete_file(path)
            except FileNotFoundError:
                pass

    def destroy_external_storage(self):
        try:
            self._fs.delete_dir(self._prefix)
        except Exception:
            logger.exception('Error cleaning up spill files. You might still have remaining spilled objects inside `{}`.'.format(self._prefix))