from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapMirrorNotFound(qexception.NotFound):
    message = _('Tap Mirror %(mirror_id)s does not exist')