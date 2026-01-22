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
class MetadefPropertyRepoProxy(NotificationRepoProxy, domain_proxy.MetadefPropertyRepo):

    def get_super_class(self):
        return domain_proxy.MetadefPropertyRepo

    def get_proxy_class(self):
        return MetadefPropertyProxy

    def get_payload(self, obj):
        return format_metadef_property_notification(obj)

    def save(self, metadef_property):
        name = getattr(metadef_property, '_old_name', metadef_property.name)
        result = super(MetadefPropertyRepoProxy, self).save(metadef_property)
        self.send_notification('metadef_property.update', metadef_property, extra_payload={'namespace': metadef_property.namespace.namespace, 'name_old': name})
        return result

    def add(self, metadef_property):
        result = super(MetadefPropertyRepoProxy, self).add(metadef_property)
        self.send_notification('metadef_property.create', metadef_property)
        return result

    def remove(self, metadef_property):
        result = super(MetadefPropertyRepoProxy, self).remove(metadef_property)
        self.send_notification('metadef_property.delete', metadef_property, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime(), 'namespace': metadef_property.namespace.namespace})
        return result