import http.client as http
import glance_store
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _LI
import glance.notifier
class ImageActionsController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None, store_api=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.store_api = store_api or glance_store
        self.gateway = glance.gateway.Gateway(self.db_api, self.store_api, self.notifier, self.policy)

    @utils.mutating
    def deactivate(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            api_pol.deactivate_image()
            status = image.status
            image.deactivate()
            if status == 'active':
                image_repo.save(image, from_state='active')
            LOG.info(_LI('Image %s is deactivated'), image_id)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to deactivate image '%s'", image_id)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.InvalidImageStatusTransition as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)

    @utils.mutating
    def reactivate(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            image = image_repo.get(image_id)
            api_pol = api_policy.ImageAPIPolicy(req.context, image, self.policy)
            api_pol.reactivate_image()
            status = image.status
            image.reactivate()
            if status == 'deactivated':
                image_repo.save(image, from_state='deactivated')
            LOG.info(_LI('Image %s is reactivated'), image_id)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to reactivate image '%s'", image_id)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.InvalidImageStatusTransition as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)