from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
class ImportmapDetailView(OperatorDetailView):
    nlri = fields.OptionalDataField('_nlri')
    rt = fields.OptionalDataField('_rt')

    def encode(self):
        ret = {}
        nlri = self.get_field('nlri')
        if nlri is not None:
            ret.update({'nlri': nlri})
        rt = self.get_field('rt')
        if rt is not None:
            ret.update({'rt': rt})
        return ret