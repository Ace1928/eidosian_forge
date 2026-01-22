from neutronclient.neutron import v2_0 as cmd_base
class ShowExt(cmd_base.ShowCommand):
    """Show information of a given resource."""
    resource = 'extension'
    allow_names = False