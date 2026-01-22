from heat.common import exception
from heat.engine import resource
from heat.engine import support
class OctaviaBase(resource.Resource):
    default_client_name = 'octavia'
    support_status = support.SupportStatus(version='10.0.0')

    def _check_status(self, expected_status='ACTIVE'):
        res = self._show_resource()
        status = res['provisioning_status']
        if status == 'ERROR':
            raise exception.ResourceInError(resource_status=status)
        return status == expected_status

    def _check_deleted(self):
        with self.client_plugin().ignore_not_found:
            return self._check_status('DELETED')
        return True

    def _resolve_attribute(self, name):
        if self.resource_id is None:
            return
        attributes = self._show_resource()
        return attributes[name]

    def handle_create(self):
        return self._prepare_args(self.properties)

    def check_create_complete(self, properties):
        if self.resource_id is None:
            try:
                res = self._resource_create(properties)
                self.resource_id_set(res['id'])
            except Exception as ex:
                if self.client_plugin().is_conflict(ex):
                    return False
                raise
        return self._check_status()

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        self._update_called = False
        return prop_diff

    def check_update_complete(self, prop_diff):
        if not prop_diff:
            return True
        if not self._update_called:
            try:
                self._resource_update(prop_diff)
                self._update_called = True
            except Exception as ex:
                if self.client_plugin().is_conflict(ex):
                    return False
                raise
        return self._check_status()

    def handle_delete(self):
        self._delete_called = False

    def check_delete_complete(self, data):
        if self.resource_id is None:
            return True
        if not self._delete_called:
            try:
                self._resource_delete()
                self._delete_called = True
            except Exception as ex:
                if self.client_plugin().is_conflict(ex):
                    return self._check_status('DELETED')
                elif self.client_plugin().is_not_found(ex):
                    return True
                raise
        return self._check_deleted()