import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
class SingleTenantConnectionManager(SwiftConnectionManager):

    def _get_storage_url(self):
        """Get swift endpoint from keystone

        Return endpoint for swift from service catalog if not overridden in
        store configuration. The method works only Keystone v3.
        If you are using different version (1 or 2)
        it returns None.
        :return: swift endpoint
        """
        if self.store.conf_endpoint:
            return self.store.conf_endpoint
        if self.store.auth_version == '3':
            try:
                return self.client.session.get_endpoint(service_type=self.store.service_type, interface=self.store.endpoint_type, region_name=self.store.region)
            except Exception as e:
                msg = _('Cannot find swift service endpoint : %s') % encodeutils.exception_to_unicode(e)
                raise exceptions.BackendException(msg)

    def _init_connection(self):
        if self.store.auth_version == '3':
            return super(SingleTenantConnectionManager, self)._init_connection()
        else:
            self.allow_reauth = False
            return self.store.get_connection(self.location, self.context)