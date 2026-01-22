from neutronclient.neutron import v2_0 as cmd_base
class ListExt(cmd_base.ListCommand):
    """List all extensions."""
    resource = 'extension'
    list_columns = ['alias', 'name']