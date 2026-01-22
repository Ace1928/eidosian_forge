from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocketsList(extension.ClientExtensionList, FoxInSocket):
    """List fox sockets."""
    shell_command = 'fox-sockets-list'
    list_columns = ['id', 'name']
    pagination_support = True
    sorting_support = True