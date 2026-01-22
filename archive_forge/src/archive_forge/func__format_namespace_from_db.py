from oslo_config import cfg
from oslo_utils import importutils
from wsme.rest import json
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common import crypt
from glance.common import exception
from glance.common import utils as common_utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _
def _format_namespace_from_db(self, namespace_obj):
    return glance.domain.MetadefNamespace(namespace_id=namespace_obj['id'], namespace=namespace_obj['namespace'], display_name=namespace_obj['display_name'], description=namespace_obj['description'], owner=namespace_obj['owner'], visibility=namespace_obj['visibility'], protected=namespace_obj['protected'], created_at=namespace_obj['created_at'], updated_at=namespace_obj['updated_at'])