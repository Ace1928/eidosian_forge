from __future__ import absolute_import, division, print_function
import os
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def upload_files(self):
    """Upload firmware and nvsram file."""
    for filename in self.add_files:
        fields = [('validate', 'true')]
        files = [('firmwareFile', filename, self.files[filename])]
        headers, data = create_multipart_formdata(files=files, fields=fields)
        try:
            rc, response = self.request('firmware/upload/', method='POST', data=data, headers=headers)
        except Exception as error:
            self.upload_failures.append(filename)
            self.module.warn('Failed to upload firmware file. File [%s]' % filename)