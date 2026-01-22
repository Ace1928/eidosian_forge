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
class MetadefResourceTypeRepoProxy(NotificationRepoProxy, domain_proxy.MetadefResourceTypeRepo):

    def get_super_class(self):
        return domain_proxy.MetadefResourceTypeRepo

    def get_proxy_class(self):
        return MetadefResourceTypeProxy

    def get_payload(self, obj):
        return format_metadef_resource_type_notification(obj)

    def add(self, md_resource_type):
        result = super(MetadefResourceTypeRepoProxy, self).add(md_resource_type)
        self.send_notification('metadef_resource_type.create', md_resource_type)
        return result

    def remove(self, md_resource_type):
        result = super(MetadefResourceTypeRepoProxy, self).remove(md_resource_type)
        self.send_notification('metadef_resource_type.delete', md_resource_type, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime(), 'namespace': md_resource_type.namespace.namespace})
        return result