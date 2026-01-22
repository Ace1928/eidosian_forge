from operator import itemgetter
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
class ExtraRouteSet(neutron.NeutronResource):
    """Resource for specifying extra routes for a Neutron router.

    Requires Neutron ``extraroute-atomic`` extension to be enabled::

      $ openstack extension show extraroute-atomic

    An extra route is a static routing table entry that is added beyond
    the routes managed implicitly by router interfaces and router gateways.

    The ``destination`` of an extra route is any IP network in /CIDR notation.
    The ``nexthop`` of an extra route is an IP in a subnet that is directly
    connected to the router.

    In a single OS::Neutron::ExtraRouteSet resource you can specify a
    set of extra routes (represented as a list) on the same virtual
    router. As an improvement over the (never formally supported)
    OS::Neutron::ExtraRoute resource this resource plugin uses a Neutron
    API extension (``extraroute-atomic``) that is not prone to race
    conditions when used to manage multiple extra routes of the same
    router. It is safe to manage multiple extra routes of the same router
    from multiple stacks.

    On the other hand use of the same route on the same router is not safe
    from multiple stacks (or between Heat and non-Heat managed Neutron extra
    routes).
    """
    support_status = support.SupportStatus(version='14.0.0')
    required_service_extension = 'extraroute-atomic'
    PROPERTIES = ROUTER, ROUTES = ('router', 'routes')
    _ROUTE_KEYS = DESTINATION, NEXTHOP = ('destination', 'nexthop')
    properties_schema = {ROUTER: properties.Schema(properties.Schema.STRING, description=_('The router id.'), required=True, constraints=[constraints.CustomConstraint('neutron.router')]), ROUTES: properties.Schema(properties.Schema.LIST, _('A set of route dictionaries for the router.'), schema=properties.Schema(properties.Schema.MAP, schema={DESTINATION: properties.Schema(properties.Schema.STRING, _('The destination network in CIDR notation.'), required=True, constraints=[constraints.CustomConstraint('net_cidr')]), NEXTHOP: properties.Schema(properties.Schema.STRING, _('The next hop for the destination.'), required=True, constraints=[constraints.CustomConstraint('ip_addr')])}), default=[], update_allowed=True)}

    def add_dependencies(self, deps):
        super(ExtraRouteSet, self).add_dependencies(deps)
        for resource in self.stack.values():
            if resource.has_interface('OS::Neutron::RouterInterface'):
                try:
                    router_id = self.properties[self.ROUTER]
                    dep_router_id = resource.properties.get(router.RouterInterface.ROUTER)
                except (ValueError, TypeError):
                    continue
                if dep_router_id == router_id:
                    deps += (self, resource)

    def handle_create(self):
        router = self.properties[self.ROUTER]
        routes = self.properties[self.ROUTES]
        _raise_if_duplicate(self.client().show_router(router), routes)
        self.client().add_extra_routes_to_router(router, {'router': {'routes': routes}})
        self.resource_id_set(router)

    def handle_delete(self):
        if not self.resource_id:
            return
        with self.client_plugin().ignore_not_found:
            self.client().remove_extra_routes_from_router(self.properties[self.ROUTER], {'router': {'routes': self.properties[self.ROUTES]}})

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        """Handle updates correctly.

        Implementing handle_update() here is not just an optimization but a
        must, because the default create/delete behavior would delete the
        unchanged part of the extra route set.
        """
        if self.ROUTES in prop_diff:
            del prop_diff[self.ROUTES]
        old = self.properties[self.ROUTES] or []
        new = json_snippet.properties(self.properties_schema)[self.ROUTES] or []
        add = _set_to_routes(_routes_to_set(new) - _routes_to_set(old))
        remove = _set_to_routes(_routes_to_set(old) - _routes_to_set(new))
        router = self.properties[self.ROUTER]
        _raise_if_duplicate(self.client().show_router(router), add)
        self.client().remove_extra_routes_from_router(router, {'router': {'routes': remove}})
        self.client().add_extra_routes_to_router(router, {'router': {'routes': add}})