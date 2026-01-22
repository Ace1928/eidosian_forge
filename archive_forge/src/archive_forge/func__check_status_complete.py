import datetime
import eventlet
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_log import log as logging
def _check_status_complete(self, started_at, wait_secs):

    def simulated_effort():
        client_name = self.properties[self.CLIENT_NAME]
        self.entity = self.properties[self.ENTITY_NAME]
        if client_name and self.entity:
            entity_id = self.data().get('value') or self.resource_id
            try:
                obj = getattr(self.client(name=client_name), self.entity)
                obj.get(entity_id)
            except Exception as exc:
                LOG.debug('%s.%s(%s) %s' % (client_name, self.entity, entity_id, str(exc)))
        else:
            eventlet.sleep(1)
    if isinstance(started_at, str):
        started_at = timeutils.parse_isotime(started_at)
    started_at = timeutils.normalize_time(started_at)
    waited = timeutils.utcnow() - started_at
    LOG.info('Resource %(name)s waited %(waited)s/%(sec)s seconds', {'name': self.name, 'waited': waited, 'sec': wait_secs})
    if wait_secs >= 0 and waited > datetime.timedelta(seconds=wait_secs):
        fail_prop = self.properties[self.FAIL]
        if fail_prop and self.action != self.DELETE:
            raise ValueError('Test Resource failed %s' % self.name)
        return True
    simulated_effort()
    return False