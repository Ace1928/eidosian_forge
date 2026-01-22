from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class PeerDetailView(OperatorDetailView):
    remote_as = fields.DataField('remote_as')
    ip_address = fields.DataField('ip_address')
    enabled = fields.DataField('enabled')
    adj_rib_in = fields.RelatedViewField('adj_rib_in', 'os_ken.services.protocols.bgp.operator.views.bgp.ReceivedRouteDictView')
    adj_rib_out = fields.RelatedViewField('adj_rib_out', 'os_ken.services.protocols.bgp.operator.views.bgp.SentRouteDictView')
    neigh_conf = fields.RelatedViewField('_neigh_conf', 'os_ken.services.protocols.bgp.operator.views.conf.ConfDetailView')
    common_conf = fields.RelatedViewField('_common_conf', 'os_ken.services.protocols.bgp.operator.views.conf.ConfDetailView')
    state = fields.RelatedViewField('state', 'os_ken.services.protocols.bgp.operator.views.bgp.PeerStateDetailView')

    def encode(self):
        ret = super(PeerDetailView, self).encode()
        ret.update({'stats': self.rel('state').encode(), 'settings': self.rel('neigh_conf').encode()})
        return ret