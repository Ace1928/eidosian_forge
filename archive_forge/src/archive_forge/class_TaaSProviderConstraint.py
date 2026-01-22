from heat.engine.clients.os.neutron import neutron_constraints as nc
class TaaSProviderConstraint(nc.ProviderConstraint):
    service_type = 'TAPASASERVICE'