from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def _create_arguments(self):
    arguments = {'size': self.properties[self.SIZE], 'availability_zone': self.properties[self.AVAILABILITY_ZONE] or None}
    scheduler_hints = self._scheduler_hints(self.properties[self.CINDER_SCHEDULER_HINTS])
    if scheduler_hints:
        arguments[self.CINDER_SCHEDULER_HINTS] = scheduler_hints
    if self.properties[self.IMAGE]:
        arguments['imageRef'] = self.client_plugin('glance').find_image_by_name_or_id(self.properties[self.IMAGE])
    elif self.properties[self.IMAGE_REF]:
        arguments['imageRef'] = self.properties[self.IMAGE_REF]
    optionals = (self.SNAPSHOT_ID, self.VOLUME_TYPE, self.SOURCE_VOLID, self.METADATA)
    arguments.update(((prop, self.properties[prop]) for prop in optionals if self.properties[prop] is not None))
    return arguments