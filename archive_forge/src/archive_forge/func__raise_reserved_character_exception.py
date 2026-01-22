from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _raise_reserved_character_exception(self, entity_type, name):
    msg = _('%(entity)s name cannot contain the following reserved characters: %(chars)s')
    raise exception.ValidationError(message=msg % {'entity': entity_type, 'chars': utils.list_url_unsafe_chars(name)})