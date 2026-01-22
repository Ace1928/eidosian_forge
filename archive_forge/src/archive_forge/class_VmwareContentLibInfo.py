from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwareContentLibInfo(VmwareRestClient):

    def __init__(self, module):
        """Constructor."""
        super(VmwareContentLibInfo, self).__init__(module)
        self.content_service = self.api_client
        self.local_content_libraries = self.content_service.content.LocalLibrary.list()
        if self.local_content_libraries is None:
            self.local_content_libraries = []
        self.subscribed_content_libraries = self.content_service.content.SubscribedLibrary.list()
        if self.subscribed_content_libraries is None:
            self.subscribed_content_libraries = []
        self.library_info = []

    def get_all_content_libs(self):
        """Method to retrieve List of content libraries."""
        content_libraries = self.local_content_libraries + self.subscribed_content_libraries
        self.module.exit_json(changed=False, content_libs=content_libraries)

    def get_content_lib_details(self, library_id):
        """Method to retrieve Details of contentlib with library_id"""
        lib_publish_info = None
        if library_id in self.local_content_libraries:
            try:
                lib_details = self.content_service.content.LocalLibrary.get(library_id)
                lib_publish_info = dict(persist_json_enabled=lib_details.publish_info.persist_json_enabled, authentication_method=lib_details.publish_info.authentication_method, publish_url=lib_details.publish_info.publish_url, published=lib_details.publish_info.published, user_name=lib_details.publish_info.user_name)
            except Exception as e:
                self.module.fail_json(exists=False, msg='%s' % self.get_error_message(e))
        elif library_id in self.subscribed_content_libraries:
            try:
                lib_details = self.content_service.content.SubscribedLibrary.get(library_id)
            except Exception as e:
                self.module.fail_json(exists=False, msg='%s' % self.get_error_message(e))
        else:
            self.module.fail_json(exists=False, msg='Library %s not found.' % library_id)
        self.library_info.append(dict(library_name=lib_details.name, library_description=lib_details.description, library_id=lib_details.id, library_type=lib_details.type, library_creation_time=lib_details.creation_time, library_server_guid=lib_details.server_guid, library_version=lib_details.version, library_publish_info=lib_publish_info))
        self.module.exit_json(exists=False, changed=False, content_lib_details=self.library_info)