import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api):
    """Setup federated username.

    Function covers all the cases for properly setting user id, a primary
    identifier for identity objects. Initial version of the mapping engine
    assumed user is identified by ``name`` and his ``id`` is built from the
    name. We, however need to be able to accept local rules that identify user
    by either id or name/domain.

    The following use-cases are covered:

    1) If neither user_name nor user_id is set raise exception.Unauthorized
    2) If user_id is set and user_name not, set user_name equal to user_id
    3) If user_id is not set and user_name is, set user_id as url safe version
       of user_name.

    Furthermore, we set the IdP as the user domain, if the user definition
    does not come with a domain definition.

    :param mapped_properties: Properties issued by a RuleProcessor.
    :type: dictionary

    :param idp_domain_id: The domain ID of the IdP registered in OpenStack.
    :type: string

    :param resource_api: The resource API used to access the database layer.
    :type: object

    :raises keystone.exception.Unauthorized: If neither `user_name` nor
        `user_id` is set.
    :returns: tuple with user identification
    :rtype: tuple

    """
    user = mapped_properties['user']
    user_id = user.get('id')
    user_name = user.get('name') or flask.request.remote_user
    if not any([user_id, user_name]):
        msg = _('Could not map user while setting ephemeral user identity. Either mapping rules must specify user id/name or REMOTE_USER environment variable must be set.')
        raise exception.Unauthorized(msg)
    elif not user_name:
        user['name'] = user_id
    elif not user_id:
        user_id = user_name
    if user_name:
        user['name'] = user_name
    user['id'] = parse.quote(user_id)
    LOG.debug('Processing domain for federated user: %s', user)
    domain = user.get('domain', {'id': idp_domain_id})
    if 'id' not in domain:
        db_domain = resource_api.get_domain_by_name(domain['name'])
        domain = {'id': db_domain.get('id')}
    user['domain'] = domain
    LOG.debug('User [%s] domain ID was resolved to [%s]', user['name'], user['domain']['id'])