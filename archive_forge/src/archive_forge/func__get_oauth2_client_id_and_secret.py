import importlib
import django.conf
from django.core import exceptions
from django.core import urlresolvers
from six.moves.urllib import parse
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
from oauth2client.contrib.django_util import storage
def _get_oauth2_client_id_and_secret(settings_instance):
    """Initializes client id and client secret based on the settings.

    Args:
        settings_instance: An instance of ``django.conf.settings``.

    Returns:
        A 2-tuple, the first item is the client id and the second
         item is the client secret.
    """
    secret_json = getattr(settings_instance, 'GOOGLE_OAUTH2_CLIENT_SECRETS_JSON', None)
    if secret_json is not None:
        return _load_client_secrets(secret_json)
    else:
        client_id = getattr(settings_instance, 'GOOGLE_OAUTH2_CLIENT_ID', None)
        client_secret = getattr(settings_instance, 'GOOGLE_OAUTH2_CLIENT_SECRET', None)
        if client_id is not None and client_secret is not None:
            return (client_id, client_secret)
        else:
            raise exceptions.ImproperlyConfigured('Must specify either GOOGLE_OAUTH2_CLIENT_SECRETS_JSON, or both GOOGLE_OAUTH2_CLIENT_ID and GOOGLE_OAUTH2_CLIENT_SECRET in settings.py')