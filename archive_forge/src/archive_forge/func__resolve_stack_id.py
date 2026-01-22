from urllib import parse
from heatclient._i18n import _
from heatclient.common import base
from heatclient.common import utils
from heatclient import exc
def _resolve_stack_id(self, stack_id):
    if stack_id.find('/') > 0:
        return stack_id
    resp = self.client.get('/stacks/%s' % stack_id, redirect=False)
    location = resp.headers.get('location')
    if not location:
        message = _('Location not returned with redirect')
        raise exc.InvalidEndpoint(message=message)
    return location.split('/stacks/', 1)[1]