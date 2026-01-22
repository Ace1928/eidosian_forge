import wsme
from wsme import types
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common.wsme_utils import WSMEModelTransformer
class MetadefObjects(types.Base, WSMEModelTransformer):
    objects = wsme.wsattr([MetadefObject], mandatory=False)
    schema = wsme.wsattr(types.text, mandatory=True)

    def __init__(self, **kwargs):
        super(MetadefObjects, self).__init__(**kwargs)