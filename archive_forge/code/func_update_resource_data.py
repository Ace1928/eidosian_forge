from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
@classmethod
def update_resource_data(cls, resource_data, status):
    if status is cls.STABLE:
        return
    if status is cls.DEPRECATED or status is cls.EXPERIMENTAL:
        resource_data['hints'] = {'status': status}
        return
    raise exception.Error(message=_('Unexpected status requested for JSON Home response, %s') % status)