import keystoneauth1.exceptions as kc_exception
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
class StackUser(resource.Resource):

    def handle_create(self):
        self._create_user()

    def _create_user(self):
        if self.data().get('user_id'):
            return
        if not self.stack.stack_user_project_id:
            project_id = self.keystone().create_stack_domain_project(self.stack.id)
            self.stack.set_stack_user_project_id(project_id)
        user_id = self.keystone().create_stack_domain_user(username=self.physical_resource_name(), password=getattr(self, 'password', None), project_id=self.stack.stack_user_project_id)
        self.data_set('user_id', user_id)

    def _user_token(self):
        project_id = self.stack.stack_user_project_id
        if not project_id:
            raise ValueError(_("Can't get user token, user not yet created"))
        password = getattr(self, 'password', None)
        if password is None:
            raise ValueError(_("Can't get user token without password"))
        return self.keystone().stack_domain_user_token(user_id=self._get_user_id(), project_id=project_id, password=password)

    def _get_user_id(self):
        user_id = self.data().get('user_id')
        if user_id:
            return user_id

    def handle_delete(self):
        self._delete_user()
        return super(StackUser, self).handle_delete()

    def _delete_user(self):
        user_id = self._get_user_id()
        if user_id is None:
            return
        self._delete_keypair()
        try:
            self.keystone().delete_stack_domain_user(user_id=user_id, project_id=self.stack.stack_user_project_id)
        except kc_exception.NotFound:
            pass
        except ValueError:
            LOG.warning('Reverting to legacy user delete path')
            try:
                self.keystone().delete_stack_user(user_id)
            except kc_exception.NotFound:
                pass
        self.data_delete('user_id')

    def handle_suspend(self):
        user_id = self._get_user_id()
        try:
            self.keystone().disable_stack_domain_user(user_id=user_id, project_id=self.stack.stack_user_project_id)
        except ValueError:
            self.keystone().disable_stack_user(user_id=user_id)

    def handle_resume(self):
        user_id = self._get_user_id()
        try:
            self.keystone().enable_stack_domain_user(user_id=user_id, project_id=self.stack.stack_user_project_id)
        except ValueError:
            self.keystone().enable_stack_user(user_id=user_id)

    def _create_keypair(self):
        if self.data().get('credential_id'):
            return
        user_id = self._get_user_id()
        kp = self.keystone().create_stack_domain_user_keypair(user_id=user_id, project_id=self.stack.stack_user_project_id)
        if not kp:
            raise exception.Error(_('Error creating ec2 keypair for user %s') % user_id)
        else:
            try:
                credential_id = kp.id
            except AttributeError:
                credential_id = kp.access
            self.data_set('credential_id', credential_id, redact=True)
            self.data_set('access_key', kp.access, redact=True)
            self.data_set('secret_key', kp.secret, redact=True)
        return kp

    def _delete_keypair(self):
        credential_id = self.data().get('credential_id')
        if not credential_id:
            return
        user_id = self._get_user_id()
        if user_id is None:
            return
        try:
            self.keystone().delete_stack_domain_user_keypair(user_id=user_id, project_id=self.stack.stack_user_project_id, credential_id=credential_id)
        except kc_exception.NotFound:
            pass
        except ValueError:
            self.keystone().delete_ec2_keypair(user_id=user_id, credential_id=credential_id)
        for data_key in ('access_key', 'secret_key', 'credential_id'):
            self.data_delete(data_key)

    def _register_access_key(self):
        """Access is limited to this resource, which created the keypair."""

        def access_allowed(resource_name):
            return resource_name == self.name
        if self.access_key is not None:
            self.stack.register_access_allowed_handler(self.access_key, access_allowed)
        if self._get_user_id() is not None:
            self.stack.register_access_allowed_handler(self._get_user_id(), access_allowed)