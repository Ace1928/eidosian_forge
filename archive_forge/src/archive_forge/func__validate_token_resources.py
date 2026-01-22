from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _validate_token_resources(self):
    if self.project and (not self.project.get('enabled')):
        msg = 'Unable to validate token because project %(id)s is disabled' % {'id': self.project_id}
        tr_msg = _('Unable to validate token because project %(id)s is disabled') % {'id': self.project_id}
        LOG.warning(msg)
        raise exception.ProjectNotFound(tr_msg)
    if self.project and (not self.project_domain.get('enabled')):
        msg = 'Unable to validate token because domain %(id)s is disabled' % {'id': self.project_domain['id']}
        tr_msg = _('Unable to validate token because domain %(id)s is disabled') % {'id': self.project_domain['id']}
        LOG.warning(msg)
        raise exception.DomainNotFound(tr_msg)