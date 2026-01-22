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
class MetadefObjectRepoProxy(NotificationRepoProxy, domain_proxy.MetadefObjectRepo):

    def get_super_class(self):
        return domain_proxy.MetadefObjectRepo

    def get_proxy_class(self):
        return MetadefObjectProxy

    def get_payload(self, obj):
        return format_metadef_object_notification(obj)

    def save(self, metadef_object):
        name = getattr(metadef_object, '_old_name', metadef_object.name)
        result = super(MetadefObjectRepoProxy, self).save(metadef_object)
        self.send_notification('metadef_object.update', metadef_object, extra_payload={'namespace': metadef_object.namespace.namespace, 'name_old': name})
        return result

    def add(self, metadef_object):
        result = super(MetadefObjectRepoProxy, self).add(metadef_object)
        self.send_notification('metadef_object.create', metadef_object)
        return result

    def remove(self, metadef_object):
        result = super(MetadefObjectRepoProxy, self).remove(metadef_object)
        self.send_notification('metadef_object.delete', metadef_object, extra_payload={'deleted': True, 'deleted_at': timeutils.isotime(), 'namespace': metadef_object.namespace.namespace})
        return result