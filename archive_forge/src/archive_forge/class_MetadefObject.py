import wsme
from wsme import types
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common.wsme_utils import WSMEModelTransformer
class MetadefObject(types.Base, WSMEModelTransformer):
    name = wsme.wsattr(types.text, mandatory=True)
    required = wsme.wsattr([types.text], mandatory=False)
    description = wsme.wsattr(types.text, mandatory=False)
    properties = wsme.wsattr({types.text: PropertyType}, mandatory=False)
    created_at = wsme.wsattr(types.text, mandatory=False)
    updated_at = wsme.wsattr(types.text, mandatory=False)
    self = wsme.wsattr(types.text, mandatory=False)
    schema = wsme.wsattr(types.text, mandatory=False)

    def __init__(cls, **kwargs):
        super(MetadefObject, cls).__init__(**kwargs)