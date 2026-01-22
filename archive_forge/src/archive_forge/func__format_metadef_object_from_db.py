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
def _format_metadef_object_from_db(self, metadata_object, namespace_entity):
    required_str = metadata_object['required']
    required_list = required_str.split(',') if required_str else []
    property_types = {}
    json_props = metadata_object['json_schema']
    for id in json_props:
        property_types[id] = json.fromjson(PropertyType, json_props[id])
    return glance.domain.MetadefObject(namespace=namespace_entity, object_id=metadata_object['id'], name=metadata_object['name'], required=required_list, description=metadata_object['description'], properties=property_types, created_at=metadata_object['created_at'], updated_at=metadata_object['updated_at'])