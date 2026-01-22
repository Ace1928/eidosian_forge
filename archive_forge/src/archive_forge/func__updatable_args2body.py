from neutronclient._i18n import _
from neutronclient.common import extension
def _updatable_args2body(parsed_args, body, client):
    if parsed_args.name:
        body['name'] = parsed_args.name