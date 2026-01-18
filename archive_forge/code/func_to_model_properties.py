import wsme
from wsme.rest import json
from wsme import types
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.common.wsme_utils import WSMEModelTransformer
@staticmethod
def to_model_properties(db_property_types):
    property_types = {}
    for db_property_type in db_property_types:
        property_type = json.fromjson(PropertyType, db_property_type.schema)
        property_type_name = db_property_type.name
        property_types[property_type_name] = property_type
    return property_types