from heat.engine.clients.os.neutron import neutron_constraints as nc
class LoadbalancerConstraint(nc.NeutronConstraint):
    resource_name = 'loadbalancer'
    extension = 'lbaasv2'