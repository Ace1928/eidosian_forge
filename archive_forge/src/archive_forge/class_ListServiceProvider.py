from neutronclient.neutron import v2_0 as neutronV20
class ListServiceProvider(neutronV20.ListCommand):
    """List service providers."""
    resource = 'service_provider'
    list_columns = ['service_type', 'name', 'default']
    _formatters = {}
    pagination_support = True
    sorting_support = True