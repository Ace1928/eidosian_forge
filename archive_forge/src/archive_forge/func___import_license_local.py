from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __import_license_local(self, import_license_url, resource_id):
    """
        Import a license locally.

        Args:
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            dict: The import status of the license.
        """
    payload = {}
    path = self.module.params.get('share_parameters').get('share_name')
    if not (os.path.exists(path) or os.path.isdir(path)):
        self.module.exit_json(msg=INVALID_DIRECTORY_MSG.format(path=path), failed=True)
    file_path = self.module.params.get('share_parameters').get('share_name') + '/' + self.module.params.get('share_parameters').get('file_name')
    file_exits = os.path.exists(file_path)
    if file_exits:
        with open(file_path, 'rb') as cert:
            cert_content = cert.read()
            read_file = base64.encodebytes(cert_content).decode('ascii')
    else:
        self.module.exit_json(msg=NO_FILE_MSG, failed=True)
    payload['LicenseFile'] = read_file
    payload['FQDD'] = resource_id
    payload['ImportOptions'] = 'Force'
    try:
        import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
    except HTTPError as err:
        filter_err = remove_key(json.load(err), regex_pattern=ODATA_REGEX)
        message_details = filter_err.get('error').get('@Message.ExtendedInfo')[0]
        message_id = message_details.get('MessageId')
        if 'LIC018' in message_id:
            self.module.exit_json(msg=message_details.get('Message'), skipped=True)
        else:
            self.module.exit_json(msg=message_details.get('Message'), error_info=filter_err, failed=True)
    return import_status