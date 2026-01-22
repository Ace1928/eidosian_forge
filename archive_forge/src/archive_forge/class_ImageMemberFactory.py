from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
class ImageMemberFactory(object):

    def new_image_member(self, image, member_id):
        created_at = timeutils.utcnow()
        updated_at = created_at
        return ImageMembership(image_id=image.image_id, member_id=member_id, created_at=created_at, updated_at=updated_at, status='pending')