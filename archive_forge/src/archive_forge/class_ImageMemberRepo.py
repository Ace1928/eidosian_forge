from oslo_config import cfg
from oslo_utils import importutils
from wsme.rest import json
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.common import crypt
from glance.common import exception
from glance.common import utils as common_utils
import glance.domain
import glance.domain.proxy
from glance.i18n import _
class ImageMemberRepo(object):

    def __init__(self, context, db_api, image):
        self.context = context
        self.db_api = db_api
        self.image = image

    def _format_image_member_from_db(self, db_image_member):
        return glance.domain.ImageMembership(id=db_image_member['id'], image_id=db_image_member['image_id'], member_id=db_image_member['member'], status=db_image_member['status'], created_at=db_image_member['created_at'], updated_at=db_image_member['updated_at'])

    def _format_image_member_to_db(self, image_member):
        image_member = {'image_id': self.image.image_id, 'member': image_member.member_id, 'status': image_member.status, 'created_at': image_member.created_at}
        return image_member

    def list(self):
        db_members = self.db_api.image_member_find(self.context, image_id=self.image.image_id)
        image_members = []
        for db_member in db_members:
            image_members.append(self._format_image_member_from_db(db_member))
        return image_members

    def add(self, image_member):
        try:
            self.get(image_member.member_id)
        except exception.NotFound:
            pass
        else:
            msg = _('The target member %(member_id)s is already associated with image %(image_id)s.') % {'member_id': image_member.member_id, 'image_id': self.image.image_id}
            raise exception.Duplicate(msg)
        image_member_values = self._format_image_member_to_db(image_member)
        members = self.db_api.image_member_find(self.context, image_id=self.image.image_id, member=image_member.member_id, include_deleted=True)
        if members:
            new_values = self.db_api.image_member_update(self.context, members[0]['id'], image_member_values)
        else:
            new_values = self.db_api.image_member_create(self.context, image_member_values)
        image_member.created_at = new_values['created_at']
        image_member.updated_at = new_values['updated_at']
        image_member.id = new_values['id']

    def remove(self, image_member):
        try:
            self.db_api.image_member_delete(self.context, image_member.id)
        except (exception.NotFound, exception.Forbidden):
            msg = _('The specified member %s could not be found')
            raise exception.NotFound(msg % image_member.id)

    def save(self, image_member, from_state=None):
        image_member_values = self._format_image_member_to_db(image_member)
        try:
            new_values = self.db_api.image_member_update(self.context, image_member.id, image_member_values)
        except (exception.NotFound, exception.Forbidden):
            raise exception.NotFound()
        image_member.updated_at = new_values['updated_at']

    def get(self, member_id):
        try:
            db_api_image_member = self.db_api.image_member_find(self.context, self.image.image_id, member_id)
            if not db_api_image_member:
                raise exception.NotFound()
        except (exception.NotFound, exception.Forbidden):
            raise exception.NotFound()
        image_member = self._format_image_member_from_db(db_api_image_member[0])
        return image_member