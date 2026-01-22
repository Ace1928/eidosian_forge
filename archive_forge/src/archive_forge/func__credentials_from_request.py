import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def _credentials_from_request(request):
    """Gets the authorized credentials for this flow, if they exist."""
    if oauth2_settings.storage_model is None or request.user.is_authenticated():
        return get_storage(request).get()
    else:
        return None