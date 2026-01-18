from oslo_config import cfg
from troveclient import client as tc
from troveclient import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def validate_datastore(self, datastore_type, datastore_version, ds_type_key, ds_version_key):
    if datastore_type:
        allowed_versions = self.client().datastore_versions.list(datastore_type)
        allowed_version_names = [v.name for v in allowed_versions]
        if datastore_version:
            if datastore_version not in allowed_version_names:
                msg = _('Datastore version %(dsversion)s for datastore type %(dstype)s is not valid. Allowed versions are %(allowed)s.') % {'dstype': datastore_type, 'dsversion': datastore_version, 'allowed': ', '.join(allowed_version_names)}
                raise exception.StackValidationFailed(message=msg)
    elif datastore_version:
        msg = _('Not allowed - %(dsver)s without %(dstype)s.') % {'dsver': ds_version_key, 'dstype': ds_type_key}
        raise exception.StackValidationFailed(message=msg)