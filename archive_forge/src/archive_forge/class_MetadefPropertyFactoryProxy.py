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
class MetadefPropertyFactoryProxy(NotificationFactoryProxy, domain_proxy.MetadefPropertyFactory):

    def get_super_class(self):
        return domain_proxy.MetadefPropertyFactory

    def get_proxy_class(self):
        return MetadefPropertyProxy