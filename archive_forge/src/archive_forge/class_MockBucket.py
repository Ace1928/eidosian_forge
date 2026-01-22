import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
class MockBucket(object):

    def __init__(self, connection=None, name=None, key_class=NOT_IMPL):
        self.name = name
        self.keys = {}
        self.acls = {name: MockAcl()}
        self.def_acl = MockAcl()
        self.subresources = {}
        self.connection = connection
        self.logging = False

    def __repr__(self):
        return 'MockBucket: %s' % self.name

    def copy_key(self, new_key_name, src_bucket_name, src_key_name, metadata=NOT_IMPL, src_version_id=NOT_IMPL, storage_class=NOT_IMPL, preserve_acl=NOT_IMPL, encrypt_key=NOT_IMPL, headers=NOT_IMPL, query_args=NOT_IMPL):
        new_key = self.new_key(key_name=new_key_name)
        src_key = self.connection.get_bucket(src_bucket_name).get_key(src_key_name)
        new_key.data = copy.copy(src_key.data)
        new_key.size = len(new_key.data)
        return new_key

    def disable_logging(self):
        self.logging = False

    def enable_logging(self, target_bucket_prefix):
        self.logging = True

    def get_logging_config(self):
        return {'Logging': {}}

    def get_versioning_status(self, headers=NOT_IMPL):
        return False

    def get_acl(self, key_name='', headers=NOT_IMPL, version_id=NOT_IMPL):
        if key_name:
            return self.acls[key_name]
        else:
            return self.acls[self.name]

    def get_def_acl(self, key_name=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        return self.def_acl

    def get_subresource(self, subresource, key_name=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        if subresource in self.subresources:
            return self.subresources[subresource]
        else:
            return '<Subresource/>'

    def get_tags(self):
        return []

    def new_key(self, key_name=None):
        mock_key = MockKey(self, key_name)
        self.keys[key_name] = mock_key
        self.acls[key_name] = MockAcl()
        return mock_key

    def delete_key(self, key_name, headers=NOT_IMPL, version_id=NOT_IMPL, mfa_token=NOT_IMPL):
        if key_name not in self.keys:
            raise boto.exception.StorageResponseError(404, 'Not Found')
        del self.keys[key_name]

    def get_all_keys(self, headers=NOT_IMPL):
        return six.itervalues(self.keys)

    def get_key(self, key_name, headers=NOT_IMPL, version_id=NOT_IMPL):
        if key_name not in self.keys:
            return None
        return self.keys[key_name]

    def list(self, prefix='', delimiter='', marker=NOT_IMPL, headers=NOT_IMPL):
        prefix = prefix or ''
        result = []
        key_name_set = set()
        for k in six.itervalues(self.keys):
            if k.name.startswith(prefix):
                k_name_past_prefix = k.name[len(prefix):]
                if delimiter:
                    pos = k_name_past_prefix.find(delimiter)
                else:
                    pos = -1
                if pos != -1:
                    key_or_prefix = Prefix(bucket=self, name=k.name[:len(prefix) + pos + 1])
                else:
                    key_or_prefix = MockKey(bucket=self, name=k.name)
                if key_or_prefix.name not in key_name_set:
                    key_name_set.add(key_or_prefix.name)
                    result.append(key_or_prefix)
        return result

    def set_acl(self, acl_or_str, key_name='', headers=NOT_IMPL, version_id=NOT_IMPL):
        if key_name:
            self.acls[key_name] = MockAcl(acl_or_str)
        else:
            self.acls[self.name] = MockAcl(acl_or_str)

    def set_def_acl(self, acl_or_str, key_name=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.def_acl = acl_or_str

    def set_subresource(self, subresource, value, key_name=NOT_IMPL, headers=NOT_IMPL, version_id=NOT_IMPL):
        self.subresources[subresource] = value