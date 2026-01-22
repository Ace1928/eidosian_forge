from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
class ImportmapManagerDetailView(OperatorDetailView):
    importmaps = fields.RelatedDictViewField('_import_maps_by_name', 'os_ken.services.protocols.bgp.operator.views.other.ImportmapDictView')