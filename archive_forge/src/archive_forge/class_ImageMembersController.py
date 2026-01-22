import copy
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
import glance.notifier
import glance.schema
class ImageMembersController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None, store_api=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.store_api = store_api or glance_store
        self.gateway = glance.gateway.Gateway(self.db_api, self.store_api, self.notifier, self.policy)

    def _get_member_repo(self, req, image):
        try:
            return self.gateway.get_member_repo(image, req.context)
        except exception.Forbidden as e:
            msg = _('Error fetching members of image %(image_id)s: %(inner_msg)s') % {'image_id': image.image_id, 'inner_msg': e.msg}
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)

    def _lookup_image(self, req, image_id):
        image_repo = self.gateway.get_repo(req.context)
        try:
            return image_repo.get(image_id)
        except exception.NotFound:
            msg = _('Image %s not found.') % image_id
            LOG.warning(msg)
            raise webob.exc.HTTPNotFound(explanation=msg)
        except exception.Forbidden:
            msg = _('You are not authorized to lookup image %s.') % image_id
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)

    def _check_visibility_and_ownership(self, context, image, ownership_check=None):
        if image.visibility != 'shared':
            message = _('Only shared images have members.')
            raise exception.Forbidden(message)
        owner = image.owner
        if not (CONF.oslo_policy.enforce_new_defaults or CONF.oslo_policy.enforce_scope) and (not context.is_admin):
            if ownership_check == 'create':
                if owner is None or owner != context.owner:
                    message = _('You are not permitted to create image members for the image.')
                    raise exception.Forbidden(message)
            elif ownership_check == 'update':
                if context.owner == owner:
                    message = _("You are not permitted to modify 'status' on this image member.")
                    raise exception.Forbidden(message)
            elif ownership_check == 'delete':
                if context.owner != owner:
                    message = _('You cannot delete image member.')
                    raise exception.Forbidden(message)

    def _lookup_member(self, req, image, member_id, member_repo=None):
        if not member_repo:
            member_repo = self._get_member_repo(req, image)
        try:
            api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy).get_member()
            return member_repo.get(member_id)
        except exception.NotFound:
            msg = _('%(m_id)s not found in the member list of the image %(i_id)s.') % {'m_id': member_id, 'i_id': image.image_id}
            LOG.warning(msg)
            raise webob.exc.HTTPNotFound(explanation=msg)
        except exception.Forbidden:
            msg = _('You are not authorized to lookup the members of the image %s.') % image.image_id
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)

    @utils.mutating
    def create(self, req, image_id, member_id):
        """
        Adds a membership to the image.
        :param req: the Request object coming from the wsgi layer
        :param image_id: the image identifier
        :param member_id: the member identifier
        :returns: The response body is a mapping of the following form

        ::

            {'member_id': <MEMBER>,
             'image_id': <IMAGE>,
             'status': <MEMBER_STATUS>
             'created_at': ..,
             'updated_at': ..}

        """
        try:
            image = self._lookup_image(req, image_id)
            self._check_visibility_and_ownership(req.context, image, ownership_check='create')
            member_repo = self._get_member_repo(req, image)
            api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy).add_member()
            image_member_factory = self.gateway.get_image_member_factory(req.context)
            new_member = image_member_factory.new_image_member(image, member_id)
            member_repo.add(new_member)
            return new_member
        except exception.Invalid as e:
            raise webob.exc.HTTPBadRequest(explanation=e.msg)
        except exception.Forbidden:
            msg = _('Not allowed to create members for image %s.') % image_id
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)
        except exception.Duplicate:
            msg = _('Member %(member_id)s is duplicated for image %(image_id)s') % {'member_id': member_id, 'image_id': image_id}
            LOG.warning(msg)
            raise webob.exc.HTTPConflict(explanation=msg)
        except exception.ImageMemberLimitExceeded as e:
            msg = _('Image member limit exceeded for image %(id)s: %(e)s:') % {'id': image_id, 'e': encodeutils.exception_to_unicode(e)}
            LOG.warning(msg)
            raise webob.exc.HTTPRequestEntityTooLarge(explanation=msg)

    @utils.mutating
    def update(self, req, image_id, member_id, status):
        """
        Update the status of a member for a given image.
        :param req: the Request object coming from the wsgi layer
        :param image_id: the image identifier
        :param member_id: the member identifier
        :param status: the status of a member
        :returns: The response body is a mapping of the following form

        ::

            {'member_id': <MEMBER>,
             'image_id': <IMAGE>,
             'status': <MEMBER_STATUS>,
             'created_at': ..,
             'updated_at': ..}

        """
        try:
            image = self._lookup_image(req, image_id)
            self._check_visibility_and_ownership(req.context, image, ownership_check='update')
            member_repo = self._get_member_repo(req, image)
            member = self._lookup_member(req, image, member_id, member_repo=member_repo)
            api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy).modify_member()
            member.status = status
            member_repo.save(member)
            return member
        except exception.Forbidden:
            msg = _('Not allowed to update members for image %s.') % image_id
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)
        except ValueError as e:
            msg = _('Incorrect request: %s') % encodeutils.exception_to_unicode(e)
            LOG.warning(msg)
            raise webob.exc.HTTPBadRequest(explanation=msg)

    def index(self, req, image_id):
        """
        Return a list of dictionaries indicating the members of the
        image, i.e., those tenants the image is shared with.

        :param req: the Request object coming from the wsgi layer
        :param image_id: The image identifier
        :returns: The response body is a mapping of the following form

        ::

            {'members': [
                {'member_id': <MEMBER>,
                 'image_id': <IMAGE>,
                 'status': <MEMBER_STATUS>,
                 'created_at': ..,
                 'updated_at': ..}, ..
            ]}

        """
        try:
            image = self._lookup_image(req, image_id)
            self._check_visibility_and_ownership(req.context, image)
            member_repo = self._get_member_repo(req, image)
            api_policy_check = api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy)
            api_policy_check.get_members()
        except exception.Forbidden as e:
            msg = _('Not allowed to list members for image %(image_id)s: %(inner_msg)s') % {'image_id': image.image_id, 'inner_msg': e.msg}
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)
        members = [member for member in member_repo.list() if api_policy_check.check('get_member')]
        return dict(members=members)

    def show(self, req, image_id, member_id):
        """
        Returns the membership of the tenant wrt to the image_id specified.

        :param req: the Request object coming from the wsgi layer
        :param image_id: The image identifier
        :returns: The response body is a mapping of the following form

        ::

            {'member_id': <MEMBER>,
             'image_id': <IMAGE>,
             'status': <MEMBER_STATUS>
             'created_at': ..,
             'updated_at': ..}

        """
        try:
            image = self._lookup_image(req, image_id)
            self._check_visibility_and_ownership(req.context, image)
            return self._lookup_member(req, image, member_id)
        except exception.Forbidden as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except webob.exc.HTTPForbidden as e:
            raise webob.exc.HTTPNotFound(explanation=e.explanation)

    @utils.mutating
    def delete(self, req, image_id, member_id):
        """
        Removes a membership from the image.
        """
        try:
            image = self._lookup_image(req, image_id)
            self._check_visibility_and_ownership(req.context, image, ownership_check='delete')
            member_repo = self._get_member_repo(req, image)
            member = self._lookup_member(req, image, member_id, member_repo=member_repo)
            api_policy.MemberAPIPolicy(req.context, image, enforcer=self.policy).delete_member()
            member_repo.remove(member)
            return webob.Response(body='', status=http.NO_CONTENT)
        except exception.Forbidden:
            msg = _('Not allowed to delete members for image %s.') % image_id
            LOG.warning(msg)
            raise webob.exc.HTTPForbidden(explanation=msg)