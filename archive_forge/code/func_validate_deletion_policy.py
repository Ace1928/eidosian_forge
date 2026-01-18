from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
@classmethod
def validate_deletion_policy(cls, policy):
    res = super(BaseVolume, cls).validate_deletion_policy(policy)
    if res:
        return res
    if policy == rsrc_defn.ResourceDefinition.SNAPSHOT and (not cfg.CONF.volumes.backups_enabled):
        msg = _('"%s" deletion policy not supported - volume backup service is not enabled.') % policy
        raise exception.StackValidationFailed(message=msg)