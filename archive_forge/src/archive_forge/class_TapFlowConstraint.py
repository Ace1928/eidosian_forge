from heat.engine.clients.os.neutron import neutron_constraints as nc
class TapFlowConstraint(nc.NeutronExtConstraint):
    resource_name = 'tap_flow'
    extension = 'taas'