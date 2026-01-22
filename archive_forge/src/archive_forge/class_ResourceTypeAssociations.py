import wsme
from wsme import types
from glance.common.wsme_utils import WSMEModelTransformer
class ResourceTypeAssociations(types.Base, WSMEModelTransformer):
    resource_type_associations = wsme.wsattr([ResourceTypeAssociation], mandatory=False)

    def __init__(self, **kwargs):
        super(ResourceTypeAssociations, self).__init__(**kwargs)