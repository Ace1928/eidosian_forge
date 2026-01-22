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
class ExportLicense(License):
    STATUS_SUCCESS = [200, 202]

    def execute(self):
        """
        Executes the export operation for a given license ID.

        :param module: The Ansible module object.
        :type module: AnsibleModule

        :return: The response from the export operation.
        :rtype: Response
        """
        share_type = self.module.params.get('share_parameters').get('share_type')
        license_id = self.module.params.get('license_id')
        self.check_license_id(license_id)
        export_license_url = self.__get_export_license_url()
        job_status = {}
        if share_type == 'local':
            export_license_response = self.__export_license_local(export_license_url)
        elif share_type in ['http', 'https']:
            export_license_response = self.__export_license_http(export_license_url)
            job_status = self.get_job_status(export_license_response)
        elif share_type == 'cifs':
            export_license_response = self.__export_license_cifs(export_license_url)
            job_status = self.get_job_status(export_license_response)
        elif share_type == 'nfs':
            export_license_response = self.__export_license_nfs(export_license_url)
            job_status = self.get_job_status(export_license_response)
        status = export_license_response.status_code
        if status in self.STATUS_SUCCESS:
            self.module.exit_json(msg=SUCCESS_EXPORT_MSG, changed=True, job_details=job_status)
        else:
            self.module.exit_json(msg=FAILURE_MSG.format(operation='export', license_id=license_id), failed=True, job_details=job_status)

    def __export_license_local(self, export_license_url):
        """
        Export the license to a local directory.

        Args:
            module (object): The Ansible module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            object: The license status after exporting.
        """
        payload = {}
        payload['EntitlementID'] = self.module.params.get('license_id')
        path = self.module.params.get('share_parameters').get('share_name')
        if not (os.path.exists(path) or os.path.isdir(path)):
            self.module.exit_json(msg=INVALID_DIRECTORY_MSG.format(path=path), failed=True)
        if not os.access(path, os.W_OK):
            self.module.exit_json(msg=INSUFFICIENT_DIRECTORY_PERMISSION_MSG.format(path=path), failed=True)
        license_name = self.module.params.get('share_parameters').get('file_name')
        if license_name:
            license_file_name = f'{license_name}_iDRAC_license.txt'
        else:
            license_file_name = f'{self.module.params['license_id']}_iDRAC_license.txt'
        license_status = self.idrac.invoke_request(export_license_url, 'POST', data=payload)
        license_data = license_status.json_data
        license_file = license_data.get('LicenseFile')
        file_name = os.path.join(path, license_file_name)
        with open(file_name, 'w') as fp:
            fp.writelines(license_file)
        return license_status

    def __export_license_http(self, export_license_url):
        """
        Export the license using the HTTP protocol.

        Args:
            module (object): The module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            str: The export status.
        """
        payload = {}
        payload['EntitlementID'] = self.module.params.get('license_id')
        proxy_details = self.get_proxy_details()
        payload.update(proxy_details)
        export_status = self.__export_license(payload, export_license_url)
        return export_status

    def __export_license_cifs(self, export_license_url):
        """
        Export the license using CIFS share type.

        Args:
            module (object): The Ansible module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            str: The export status.
        """
        payload = {}
        payload['EntitlementID'] = self.module.params.get('license_id')
        payload['ShareType'] = 'CIFS'
        if self.module.params.get('share_parameters').get('workgroup'):
            payload['Workgroup'] = self.module.params.get('share_parameters').get('workgroup')
        share_details = self.get_share_details()
        payload.update(share_details)
        export_status = self.__export_license(payload, export_license_url)
        return export_status

    def __export_license_nfs(self, export_license_url):
        """
        Export the license using NFS share type.

        Args:
            module (object): The Ansible module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            dict: The export status of the license.
        """
        payload = {}
        payload['EntitlementID'] = self.module.params.get('license_id')
        payload['ShareType'] = 'NFS'
        payload['IPAddress'] = self.module.params.get('share_parameters').get('ip_address')
        payload['ShareName'] = self.module.params.get('share_parameters').get('share_name')
        export_status = self.__export_license(payload, export_license_url)
        return export_status

    def __get_export_license_url(self):
        """
        Get the export license URL.

        :param module: The module object.
        :type module: object
        :return: The export license URL.
        :rtype: str
        """
        uri, error_msg = validate_and_get_first_resource_id_uri(self.module, self.idrac, MANAGERS_URI)
        if error_msg:
            self.module.exit_json(msg=error_msg, failed=True)
        resp = get_dynamic_uri(self.idrac, uri)
        url = resp.get('Links', {}).get(OEM, {}).get(MANUFACTURER, {}).get(LICENSE_MANAGEMENT_SERVICE, {}).get(ODATA, {})
        action_resp = get_dynamic_uri(self.idrac, url)
        license_service = EXPORT_LOCAL if self.module.params.get('share_parameters').get('share_type') == 'local' else EXPORT_NETWORK_SHARE
        export_url = action_resp.get(ACTIONS, {}).get(license_service, {}).get('target', {})
        return export_url

    def __export_license(self, payload, export_license_url):
        """
        Export the license to a file.

        Args:
            module (object): The Ansible module object.
            payload (dict): The payload containing the license information.
            export_license_url (str): The URL for exporting the license.

        Returns:
            dict: The license status after exporting.
        """
        license_name = self.module.params.get('share_parameters').get('file_name')
        if license_name:
            license_file_name = f'{license_name}_iDRAC_license.xml'
        else:
            license_file_name = f'{self.module.params['license_id']}_iDRAC_license.xml'
        payload['FileName'] = license_file_name
        license_status = self.idrac.invoke_request(export_license_url, 'POST', data=payload)
        return license_status