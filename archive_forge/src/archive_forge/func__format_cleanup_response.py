from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
def _format_cleanup_response(cleaning, unavailable):
    column_headers = ('ID', 'Cluster Name', 'Host', 'Binary', 'Status')
    combined_data = []
    for obj in cleaning:
        details = (obj.id, obj.cluster_name, obj.host, obj.binary, 'Cleaning')
        combined_data.append(details)
    for obj in unavailable:
        details = (obj.id, obj.cluster_name, obj.host, obj.binary, 'Unavailable')
        combined_data.append(details)
    return (column_headers, combined_data)