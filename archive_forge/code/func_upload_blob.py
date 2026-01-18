from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
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