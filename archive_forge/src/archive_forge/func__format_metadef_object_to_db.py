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
def _format_metadef_object_to_db(self, metadata_object):
    required_str = ','.join(metadata_object.required) if metadata_object.required else None
    properties = metadata_object.properties
    db_schema = {}
    if properties:
        for k, v in properties.items():
            json_data = json.tojson(PropertyType, v)
            db_schema[k] = json_data
    db_metadata_object = {'name': metadata_object.name, 'required': required_str, 'description': metadata_object.description, 'json_schema': db_schema}
    return db_metadata_object