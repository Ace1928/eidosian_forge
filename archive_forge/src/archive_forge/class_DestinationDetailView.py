from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import \
from os_ken.services.protocols.bgp.operator.views.base import OperatorDetailView
from os_ken.services.protocols.bgp.operator.views import fields
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
class DestinationDetailView(OperatorDetailView):
    table = fields.RelatedViewField('_table', 'os_ken.services.protocols.bgp.operator.views.bgp.TableDetailView')
    best_path = fields.RelatedViewField('best_path', 'os_ken.services.protocols.bgp.operator.views.bgp.PathDetailView')
    known_path_list = fields.RelatedListViewField('known_path_list', 'os_ken.services.protocols.bgp.operator.views.bgp.PathListView')
    new_path_list = fields.RelatedListViewField('_new_path_list', 'os_ken.services.protocols.bgp.operator.views.bgp.PathListView')
    withdraw_list = fields.RelatedListViewField('_withdraw_list', 'os_ken.services.protocols.bgp.operator.views.bgp.PathListView')
    sent_routes = fields.RelatedListViewField('sent_routes', 'os_ken.services.protocols.bgp.operator.views.bgp.SentRouteListView')
    nlri = fields.DataField('nlri')
    route_family = fields.DataField('route_family')