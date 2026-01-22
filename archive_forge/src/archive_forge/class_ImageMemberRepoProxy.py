import abc
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import encodeutils
from oslo_utils import excutils
import webob
from glance.common import exception
from glance.common import timeutils
from glance.domain import proxy as domain_proxy
from glance.i18n import _, _LE
class ImageMemberRepoProxy(NotificationBase, domain_proxy.MemberRepo):

    def __init__(self, repo, image, context, notifier):
        self.repo = repo
        self.image = image
        self.context = context
        self.notifier = notifier
        proxy_kwargs = {'context': self.context, 'notifier': self.notifier}
        proxy_class = self.get_proxy_class()
        super_class = self.get_super_class()
        super_class.__init__(self, image, repo, proxy_class, proxy_kwargs)

    def get_super_class(self):
        return domain_proxy.MemberRepo

    def get_proxy_class(self):
        return ImageMemberProxy

    def get_payload(self, obj):
        return format_image_member_notification(obj)

    def save(self, member, from_state=None):
        super(ImageMemberRepoProxy, self).save(member, from_state=from_state)
        self.send_notification('image.member.update', member)

    def add(self, member):
        super(ImageMemberRepoProxy, self).add(member)
        self.send_notification('image.member.create', member)

    def remove(self, member):
        super(ImageMemberRepoProxy, self).remove(member)
        self.send_notification('image.member.delete', member, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime()})