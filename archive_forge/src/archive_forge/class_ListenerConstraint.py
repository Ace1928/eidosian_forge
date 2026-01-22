from heat.engine.clients.os.neutron import neutron_constraints as nc
class ListenerConstraint(nc.NeutronConstraint):
    resource_name = 'listener'
    extension = 'lbaasv2'