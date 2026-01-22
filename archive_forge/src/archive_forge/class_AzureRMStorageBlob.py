from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMStorageBlob(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(storage_account_name=dict(required=True, type='str', aliases=['account_name', 'storage_account']), blob=dict(type='str', aliases=['blob_name']), blob_type=dict(type='str', default='block', choices=['block', 'page']), container=dict(required=True, type='str', aliases=['container_name']), dest=dict(type='path', aliases=['destination']), force=dict(type='bool', default=False), resource_group=dict(required=True, type='str', aliases=['resource_group_name']), src=dict(type='str', aliases=['source']), batch_upload_src=dict(type='path'), batch_upload_dst=dict(type='path'), state=dict(type='str', default='present', choices=['absent', 'present']), public_access=dict(type='str', choices=['container', 'blob']), content_type=dict(type='str'), content_encoding=dict(type='str'), content_language=dict(type='str'), content_disposition=dict(type='str'), cache_control=dict(type='str'), content_md5=dict(type='str'))
        mutually_exclusive = [('src', 'dest'), ('src', 'batch_upload_src'), ('dest', 'batch_upload_src')]
        self.blob_service_client = None
        self.blob_details = None
        self.storage_account_name = None
        self.blob = None
        self.blob_obj = None
        self.blob_type = None
        self.container = None
        self.container_obj = None
        self.dest = None
        self.force = None
        self.resource_group = None
        self.src = None
        self.batch_upload_src = None
        self.batch_upload_dst = None
        self.state = None
        self.tags = None
        self.public_access = None
        self.results = dict(changed=False, actions=[], container=dict(), blob=dict())
        super(AzureRMStorageBlob, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, mutually_exclusive=mutually_exclusive, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results['check_mode'] = self.check_mode
        self.blob_service_client = self.get_blob_service_client(self.resource_group, self.storage_account_name)
        self.container_obj = self.get_container()
        if self.blob:
            self.blob_obj = self.get_blob()
        if self.state == 'present':
            if not self.container_obj:
                self.create_container()
            elif self.container_obj and (not self.blob):
                update_tags, self.container_obj['tags'] = self.update_tags(self.container_obj.get('tags'))
                if update_tags:
                    self.update_container_tags(self.container_obj['tags'])
            if self.batch_upload_src:
                self.batch_upload()
                return self.results
            if self.blob:
                if self.src and self.src_is_valid():
                    if self.blob_obj and (not self.force):
                        self.log('Cannot upload to {0}. Blob with that name already exists. Use the force option'.format(self.blob))
                    else:
                        self.upload_blob()
                elif self.dest and self.dest_is_valid():
                    self.download_blob()
                update_tags, self.blob_obj['tags'] = self.update_tags(self.blob_obj.get('tags'))
                if update_tags:
                    self.update_blob_tags(self.blob_obj['tags'])
                if self.blob_content_settings_differ():
                    self.update_blob_content_settings()
        elif self.state == 'absent':
            if self.container_obj and (not self.blob):
                if self.container_has_blobs():
                    if self.force:
                        self.delete_container()
                    else:
                        self.log('Cannot delete container {0}. It contains blobs. Use the force option.'.format(self.container))
                else:
                    self.delete_container()
            elif self.container_obj and self.blob_obj:
                self.delete_blob()
        del self.results['actions']
        return self.results

    def batch_upload(self):

        def _glob_files_locally(folder_path):
            len_folder_path = len(folder_path) + 1
            for root, v, files in os.walk(folder_path):
                for f in files:
                    full_path = os.path.join(root, f)
                    yield (full_path, full_path[len_folder_path:])

        def _normalize_blob_file_path(path, name):
            path_sep = '/'
            if path:
                name = path_sep.join((path, name))
            return path_sep.join(os.path.normpath(name).split(os.path.sep)).strip(path_sep)

        def _guess_content_type(file_path, original):
            if original.content_encoding or original.content_type:
                return original
            mimetypes.add_type('application/json', '.json')
            mimetypes.add_type('application/javascript', '.js')
            mimetypes.add_type('application/wasm', '.wasm')
            content_type, v = mimetypes.guess_type(file_path)
            return ContentSettings(content_type=content_type, content_disposition=original.content_disposition, content_language=original.content_language, content_md5=original.content_md5, cache_control=original.cache_control)
        if not os.path.exists(self.batch_upload_src):
            self.fail('batch upload source source directory {0} does not exist'.format(self.batch_upload_src))
        if not os.path.isdir(self.batch_upload_src):
            self.fail('incorrect usage: {0} is not a directory'.format(self.batch_upload_src))
        source_dir = os.path.realpath(self.batch_upload_src)
        source_files = list(_glob_files_locally(source_dir))
        content_settings = ContentSettings(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=None)
        for src, blob_path in source_files:
            if self.batch_upload_dst:
                blob_path = _normalize_blob_file_path(self.batch_upload_dst, blob_path)
            if not self.check_mode:
                try:
                    client = self.blob_service_client.get_blob_client(container=self.container, blob=blob_path)
                    with open(src, 'rb') as data:
                        client.upload_blob(data=data, blob_type=self.get_blob_type(self.blob_type), metadata=self.tags, content_settings=_guess_content_type(src, content_settings), overwrite=self.force)
                except Exception as exc:
                    self.fail('Error creating blob {0} - {1}'.format(src, str(exc)))
            self.results['actions'].append('created blob from {0}'.format(src))
        self.results['changed'] = True
        self.results['container'] = self.container_obj

    def get_blob_type(self, blob_type):
        if blob_type == 'block':
            return BlobType.BlockBlob
        elif blob_type == 'page':
            return BlobType.PageBlob
        else:
            return BlobType.AppendBlob

    def get_container(self):
        result = {}
        container = None
        if self.container:
            try:
                container = self.blob_service_client.get_container_client(container=self.container).get_container_properties()
            except ResourceNotFoundError:
                pass
        if container:
            result = dict(name=container['name'], tags=container['metadata'], last_modified=container['last_modified'].strftime('%d-%b-%Y %H:%M:%S %z'))
        return result

    def get_blob(self):
        result = dict()
        blob = None
        if self.blob:
            try:
                blob = self.blob_service_client.get_blob_client(container=self.container, blob=self.blob).get_blob_properties()
            except ResourceNotFoundError:
                pass
        if blob:
            result = dict(name=blob['name'], tags=blob['metadata'], last_modified=blob['last_modified'].strftime('%d-%b-%Y %H:%M:%S %z'), type=blob['blob_type'], content_length=blob['size'], content_settings=dict(content_type=blob['content_settings']['content_type'], content_encoding=blob['content_settings']['content_encoding'], content_language=blob['content_settings']['content_language'], content_disposition=blob['content_settings']['content_disposition'], cache_control=blob['content_settings']['cache_control'], content_md5=blob['content_settings']['content_md5'].hex() if blob['content_settings']['content_md5'] else None))
        return result

    def create_container(self):
        self.log('Create container %s' % self.container)
        tags = None
        if not self.blob and self.tags:
            tags = self.tags
        if not self.check_mode:
            try:
                client = self.blob_service_client.get_container_client(container=self.container)
                client.create_container(metadata=tags, public_access=self.public_access)
            except Exception as exc:
                self.fail('Error creating container {0} - {1}'.format(self.container, str(exc)))
        self.container_obj = self.get_container()
        self.results['changed'] = True
        self.results['actions'].append('created container {0}'.format(self.container))
        self.results['container'] = self.container_obj

    def upload_blob(self):
        content_settings = None
        if self.content_type or self.content_encoding or self.content_language or self.content_disposition or self.cache_control or self.content_md5:
            content_settings = ContentSettings(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=self.content_md5)
        if not self.check_mode:
            try:
                client = self.blob_service_client.get_blob_client(container=self.container, blob=self.blob)
                with open(self.src, 'rb') as data:
                    client.upload_blob(data=data, blob_type=self.get_blob_type(self.blob_type), metadata=self.tags, content_settings=content_settings, overwrite=self.force)
            except Exception as exc:
                self.fail('Error creating blob {0} - {1}'.format(self.blob, str(exc)))
        self.blob_obj = self.get_blob()
        self.results['changed'] = True
        self.results['actions'].append('created blob {0} from {1}'.format(self.blob, self.src))
        self.results['container'] = self.container_obj
        self.results['blob'] = self.blob_obj

    def download_blob(self):
        if not self.check_mode:
            try:
                client = self.blob_service_client.get_blob_client(container=self.container, blob=self.blob)
                with open(self.dest, 'wb') as blob_stream:
                    blob_data = client.download_blob()
                    blob_data.readinto(blob_stream)
            except Exception as exc:
                self.fail('Failed to download blob {0}:{1} to {2} - {3}'.format(self.container, self.blob, self.dest, exc))
        self.results['changed'] = True
        self.results['actions'].append('downloaded blob {0}:{1} to {2}'.format(self.container, self.blob, self.dest))
        self.results['container'] = self.container_obj
        self.results['blob'] = self.blob_obj

    def src_is_valid(self):
        if not os.path.isfile(self.src):
            self.fail('The source path must be a file.')
        if os.access(self.src, os.R_OK):
            return True
        self.fail('Failed to access {0}. Make sure the file exists and that you have read access.'.format(self.src))

    def dest_is_valid(self):
        if not self.check_mode:
            if not os.path.basename(self.dest):
                if os.path.isdir(self.dest):
                    self.log('Path is dir. Appending blob name.')
                    self.dest += self.blob
                else:
                    try:
                        self.log('Attempting to makedirs {0}'.format(self.dest))
                        os.makedirs(self.dest)
                    except IOError as exc:
                        self.fail('Failed to create directory {0} - {1}'.format(self.dest, str(exc)))
                    self.dest += self.blob
            else:
                file_name = os.path.basename(self.dest)
                path = self.dest.replace(file_name, '')
                self.log('Checking path {0}'.format(path))
                if not os.path.isdir(path):
                    try:
                        self.log('Attempting to makedirs {0}'.format(path))
                        os.makedirs(path)
                    except IOError as exc:
                        self.fail('Failed to create directory {0} - {1}'.format(path, str(exc)))
            self.log('Checking final path {0}'.format(self.dest))
            if os.path.isfile(self.dest) and (not self.force):
                self.log('Dest {0} already exists. Cannot download. Use the force option.'.format(self.dest))
                return False
        return True

    def delete_container(self):
        if not self.check_mode:
            try:
                self.blob_service_client.get_container_client(container=self.container).delete_container()
            except Exception as exc:
                self.fail('Error deleting container {0} - {1}'.format(self.container, str(exc)))
        self.results['changed'] = True
        self.results['actions'].append('deleted container {0}'.format(self.container))

    def container_has_blobs(self):
        try:
            blobs = self.blob_service_client.get_container_client(container=self.container).list_blobs()
        except Exception as exc:
            self.fail('Error list blobs in {0} - {1}'.format(self.container, str(exc)))
        if len(list(blobs)) > 0:
            return True
        return False

    def delete_blob(self):
        if not self.check_mode:
            try:
                self.blob_service_client.get_container_client(container=self.container).delete_blob(blob=self.blob)
            except Exception as exc:
                self.fail('Error deleting blob {0}:{1} - {2}'.format(self.container, self.blob, str(exc)))
        self.results['changed'] = True
        self.results['actions'].append('deleted blob {0}:{1}'.format(self.container, self.blob))
        self.results['container'] = self.container_obj

    def update_container_tags(self, tags):
        if not self.check_mode:
            try:
                self.blob_service_client.get_container_client(container=self.container).set_container_metadata(metadata=tags)
            except Exception as exc:
                self.fail('Error updating container tags {0} - {1}'.format(self.container, str(exc)))
        self.container_obj = self.get_container()
        self.results['changed'] = True
        self.results['actions'].append('updated container {0} tags.'.format(self.container))
        self.results['container'] = self.container_obj

    def update_blob_tags(self, tags):
        if not self.check_mode:
            try:
                self.blob_service_client.get_blob_client(container=self.container, blob=self.blob).set_blob_metadata(metadata=tags)
            except Exception as exc:
                self.fail('Update blob tags {0}:{1} - {2}'.format(self.container, self.blob, str(exc)))
        self.blob_obj = self.get_blob()
        self.results['changed'] = True
        self.results['actions'].append('updated blob {0}:{1} tags.'.format(self.container, self.blob))
        self.results['container'] = self.container_obj
        self.results['blob'] = self.blob_obj

    def blob_content_settings_differ(self):
        if self.content_type or self.content_encoding or self.content_language or self.content_disposition or self.cache_control or self.content_md5:
            settings = dict(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=self.content_md5)
            if self.blob_obj['content_settings'] != settings:
                return True
        return False

    def update_blob_content_settings(self):
        content_settings = ContentSettings(content_type=self.content_type, content_encoding=self.content_encoding, content_language=self.content_language, content_disposition=self.content_disposition, cache_control=self.cache_control, content_md5=self.content_md5)
        if not self.check_mode:
            try:
                self.blob_service_client.get_blob_client(container=self.container, blob=self.blob).set_http_headers(content_settings=content_settings)
            except Exception as exc:
                self.fail('Update blob content settings {0}:{1} - {2}'.format(self.container, self.blob, str(exc)))
        self.blob_obj = self.get_blob()
        self.results['changed'] = True
        self.results['actions'].append('updated blob {0}:{1} content settings.'.format(self.container, self.blob))
        self.results['container'] = self.container_obj
        self.results['blob'] = self.blob_obj