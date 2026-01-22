from io import BytesIO
from .... import config, errors, osutils, tests
from .... import transport as _mod_transport
from ... import netrc_credential_store
def _get_netrc_cs(self):
    return config.credential_store_registry.get_credential_store('netrc')