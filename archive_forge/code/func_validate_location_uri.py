import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
def validate_location_uri(location):
    """Validate location uri into acceptable format.

    :param location: Location uri to be validated
    """
    if not location:
        raise exception.BadStoreUri(_('Invalid location: %s') % location)
    elif location.startswith(('http://', 'https://')):
        return location
    elif location.startswith(('file:///', 'filesystem:///')):
        msg = _('File based imports are not allowed. Please use a non-local source of image data.')
        raise exception.BadStoreUri(msg)
    else:
        supported = ['http']
        msg = _('The given uri is not valid. Please specify a valid uri from the following list of supported uri %(supported)s') % {'supported': supported}
        raise urllib.error.URLError(msg)