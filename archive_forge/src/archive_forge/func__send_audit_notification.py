import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _send_audit_notification(action, initiator, outcome, target, event_type, reason=None, **kwargs):
    """Send CADF notification to inform observers about the affected resource.

    This method logs an exception when sending the notification fails.

    :param action: CADF action being audited (e.g., 'authenticate')
    :param initiator: CADF resource representing the initiator
    :param outcome: The CADF outcome (taxonomy.OUTCOME_PENDING,
        taxonomy.OUTCOME_SUCCESS, taxonomy.OUTCOME_FAILURE)
    :param target: CADF resource representing the target
    :param event_type: An OpenStack-ism, typically this is the meter name that
        Ceilometer uses to poll events.
    :param kwargs: Any additional arguments passed in will be added as
        key-value pairs to the CADF event.
    :param reason: Reason for the notification which contains the response
        code and message description
    """
    if _check_notification_opt_out(event_type, outcome):
        return
    global _CATALOG_HELPER_OBJ
    if _CATALOG_HELPER_OBJ is None:
        _CATALOG_HELPER_OBJ = _CatalogHelperObj()
    service_list = _CATALOG_HELPER_OBJ.catalog_api.list_services()
    service_id = None
    for i in service_list:
        if i['type'] == SERVICE:
            service_id = i['id']
            break
    initiator = _add_username_to_initiator(initiator)
    event = eventfactory.EventFactory().new_event(eventType=cadftype.EVENTTYPE_ACTIVITY, outcome=outcome, action=action, initiator=initiator, target=target, reason=reason, observer=resource.Resource(typeURI=taxonomy.SERVICE_SECURITY))
    if service_id is not None:
        event.observer.id = service_id
    for key, value in kwargs.items():
        setattr(event, key, value)
    context = {}
    payload = event.as_dict()
    notifier = _get_notifier()
    if notifier:
        try:
            notifier.info(context, event_type, payload)
        except Exception:
            LOG.exception('Failed to send %(action)s %(event_type)s notification', {'action': action, 'event_type': event_type})