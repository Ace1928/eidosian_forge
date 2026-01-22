from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
class NetAppElementSWAdminUser(object):
    """
    Class to set, modify and delete admin users on ElementSW box
    """

    def __init__(self):
        """
        Initialize the NetAppElementSWAdminUser class.
        """
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), element_username=dict(required=True, type='str'), element_password=dict(required=False, type='str', no_log=True), acceptEula=dict(required=False, type='bool'), access=dict(required=False, type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        param = self.module.params
        self.state = param['state']
        self.element_username = param['element_username']
        self.element_password = param['element_password']
        self.acceptEula = param['acceptEula']
        self.access = param['access']
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)
        self.elementsw_helper = NaElementSWModule(self.sfe)
        self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_admin_users')

    def does_admin_user_exist(self):
        """
        Checks to see if an admin user exists or not
        :return: True if the user exist, False if it dose not exist
        """
        admins_list = self.sfe.list_cluster_admins()
        for admin in admins_list.cluster_admins:
            if admin.username == self.element_username:
                return True
        return False

    def get_admin_user(self):
        """
        Get the admin user object
        :return: the admin user object
        """
        admins_list = self.sfe.list_cluster_admins()
        for admin in admins_list.cluster_admins:
            if admin.username == self.element_username:
                return admin
        return None

    def modify_admin_user(self):
        """
        Modify a admin user. If a password is set the user will be modified as there is no way to
        compare a new password with an existing one
        :return: if a user was modified or not
        """
        changed = False
        admin_user = self.get_admin_user()
        if self.access is not None and len(self.access) > 0:
            for access in self.access:
                if access not in admin_user.access:
                    changed = True
        if changed and (not self.module.check_mode):
            self.sfe.modify_cluster_admin(cluster_admin_id=admin_user.cluster_admin_id, access=self.access, password=self.element_password, attributes=self.attributes)
        return changed

    def add_admin_user(self):
        """
        Add's a new admin user to the element cluster
        :return: nothing
        """
        self.sfe.add_cluster_admin(username=self.element_username, password=self.element_password, access=self.access, accept_eula=self.acceptEula, attributes=self.attributes)

    def delete_admin_user(self):
        """
        Deletes an existing admin user from the element cluster
        :return: nothing
        """
        admin_user = self.get_admin_user()
        self.sfe.remove_cluster_admin(cluster_admin_id=admin_user.cluster_admin_id)

    def apply(self):
        """
        determines which method to call to set, delete or modify admin users
        :return:
        """
        changed = False
        if self.state == 'present':
            if self.does_admin_user_exist():
                changed = self.modify_admin_user()
            else:
                if not self.module.check_mode:
                    self.add_admin_user()
                changed = True
        elif self.does_admin_user_exist():
            if not self.module.check_mode:
                self.delete_admin_user()
            changed = True
        self.module.exit_json(changed=changed)