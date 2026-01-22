import http.client as http
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import webob.exc
from wsme.rest import json
from glance.api import policy
from glance.api.v2.model.metadef_namespace import Namespace
from glance.api.v2.model.metadef_namespace import Namespaces
from glance.api.v2.model.metadef_object import MetadefObject
from glance.api.v2.model.metadef_property_type import PropertyType
from glance.api.v2.model.metadef_resource_type import ResourceTypeAssociation
from glance.api.v2.model.metadef_tag import MetadefTag
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance.common import wsme_utils
import glance.db
import glance.gateway
from glance.i18n import _, _LE
import glance.notifier
import glance.schema
class NamespaceController(object):

    def __init__(self, db_api=None, policy_enforcer=None, notifier=None):
        self.db_api = db_api or glance.db.get_api()
        self.policy = policy_enforcer or policy.Enforcer()
        self.notifier = notifier or glance.notifier.Notifier()
        self.gateway = glance.gateway.Gateway(db_api=self.db_api, notifier=self.notifier, policy_enforcer=self.policy)
        self.ns_schema_link = '/v2/schemas/metadefs/namespace'
        self.obj_schema_link = '/v2/schemas/metadefs/object'
        self.tag_schema_link = '/v2/schemas/metadefs/tag'

    def index(self, req, marker=None, limit=None, sort_key='created_at', sort_dir='desc', filters=None):
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            policy_check = api_policy.MetadefAPIPolicy(req.context, enforcer=self.policy)
            policy_check.get_metadef_namespaces()
            policy_check.list_metadef_resource_types()
            if marker:
                namespace_obj = ns_repo.get(marker)
                marker = namespace_obj.namespace_id
            database_ns_list = ns_repo.list(marker=marker, limit=limit, sort_key=sort_key, sort_dir=sort_dir, filters=filters)
            ns_list = [ns for ns in database_ns_list if api_policy.MetadefAPIPolicy(req.context, md_resource=ns, enforcer=self.policy).check('get_metadef_namespace')]
            rs_repo = self.gateway.get_metadef_resource_type_repo(req.context)
            for db_namespace in ns_list:
                filters = dict()
                filters['namespace'] = db_namespace.namespace
                try:
                    repo_rs_type_list = rs_repo.list(filters=filters)
                except exception.NotFound:
                    repo_rs_type_list = []
                resource_type_list = [ResourceTypeAssociation.to_wsme_model(resource_type) for resource_type in repo_rs_type_list]
                if resource_type_list:
                    db_namespace.resource_type_associations = resource_type_list
            namespace_list = [Namespace.to_wsme_model(db_namespace, get_namespace_href(db_namespace), self.ns_schema_link) for db_namespace in ns_list]
            namespaces = Namespaces()
            namespaces.namespaces = namespace_list
            if len(namespace_list) != 0 and len(namespace_list) == limit:
                namespaces.next = ns_list[-1].namespace
        except exception.Forbidden as e:
            LOG.debug('User not permitted to retrieve metadata namespaces index')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return namespaces

    @utils.mutating
    def create(self, req, namespace):
        try:
            namespace_created = False
            ns_factory = self.gateway.get_metadef_namespace_factory(req.context)
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            policy_check = api_policy.MetadefAPIPolicy(req.context, enforcer=self.policy)
            policy_check.add_metadef_namespace()
            if namespace.resource_type_associations:
                policy_check.add_metadef_resource_type_association()
            if namespace.objects:
                policy_check.add_metadef_object()
            if namespace.properties:
                policy_check.add_metadef_property()
            if namespace.tags:
                policy_check.add_metadef_tag()
            kwargs = namespace.to_dict()
            if 'owner' not in kwargs:
                kwargs.update({'owner': req.context.owner})
            new_namespace = ns_factory.new_namespace(**kwargs)
            ns_repo.add(new_namespace)
            namespace_created = True
            if namespace.resource_type_associations:
                rs_factory = self.gateway.get_metadef_resource_type_factory(req.context)
                rs_repo = self.gateway.get_metadef_resource_type_repo(req.context)
                for resource_type in namespace.resource_type_associations:
                    new_resource = rs_factory.new_resource_type(namespace=namespace.namespace, **resource_type.to_dict())
                    rs_repo.add(new_resource)
            if namespace.objects:
                object_factory = self.gateway.get_metadef_object_factory(req.context)
                object_repo = self.gateway.get_metadef_object_repo(req.context)
                for metadata_object in namespace.objects:
                    new_meta_object = object_factory.new_object(namespace=namespace.namespace, **metadata_object.to_dict())
                    object_repo.add(new_meta_object)
            if namespace.tags:
                tag_factory = self.gateway.get_metadef_tag_factory(req.context)
                tag_repo = self.gateway.get_metadef_tag_repo(req.context)
                for metadata_tag in namespace.tags:
                    new_meta_tag = tag_factory.new_tag(namespace=namespace.namespace, **metadata_tag.to_dict())
                    tag_repo.add(new_meta_tag)
            if namespace.properties:
                prop_factory = self.gateway.get_metadef_property_factory(req.context)
                prop_repo = self.gateway.get_metadef_property_repo(req.context)
                for name, value in namespace.properties.items():
                    new_property_type = prop_factory.new_namespace_property(namespace=namespace.namespace, **self._to_property_dict(name, value))
                    prop_repo.add(new_property_type)
        except exception.Invalid as e:
            msg = _("Couldn't create metadata namespace: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Forbidden as e:
            self._cleanup_namespace(ns_repo, namespace, namespace_created)
            LOG.debug('User not permitted to create metadata namespace')
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            self._cleanup_namespace(ns_repo, namespace, namespace_created)
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            self._cleanup_namespace(ns_repo, namespace, namespace_created)
            raise webob.exc.HTTPConflict(explanation=e.msg)
        new_namespace.properties = namespace.properties
        new_namespace.objects = namespace.objects
        new_namespace.resource_type_associations = namespace.resource_type_associations
        new_namespace.tags = namespace.tags
        return Namespace.to_wsme_model(new_namespace, get_namespace_href(new_namespace), self.ns_schema_link)

    def _to_property_dict(self, name, value):
        db_property_type_dict = dict()
        db_property_type_dict['schema'] = json.tojson(PropertyType, value)
        db_property_type_dict['name'] = name
        return db_property_type_dict

    def _cleanup_namespace(self, namespace_repo, namespace, namespace_created):
        if namespace_created:
            try:
                namespace_obj = namespace_repo.get(namespace.namespace)
                namespace_obj.delete()
                namespace_repo.remove(namespace_obj)
                LOG.debug('Cleaned up namespace %(namespace)s ', {'namespace': namespace.namespace})
            except Exception as e:
                msg = (_LE('Failed to delete namespace %(namespace)s.Exception: %(exception)s'), {'namespace': namespace.namespace, 'exception': encodeutils.exception_to_unicode(e)})
                LOG.error(msg)

    def show(self, req, namespace, filters=None):
        try:
            ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
            try:
                namespace_obj = ns_repo.get(namespace)
                policy_check = api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy)
                policy_check.get_metadef_namespace()
            except (exception.Forbidden, webob.exc.HTTPForbidden):
                LOG.debug("User not permitted to show namespace '%s'", namespace)
                raise webob.exc.HTTPNotFound()
            policy_check.list_metadef_resource_types()
            policy_check.get_metadef_objects()
            policy_check.get_metadef_properties()
            policy_check.get_metadef_tags()
            namespace_detail = Namespace.to_wsme_model(namespace_obj, get_namespace_href(namespace_obj), self.ns_schema_link)
            ns_filters = dict()
            ns_filters['namespace'] = namespace
            object_repo = self.gateway.get_metadef_object_repo(req.context)
            db_metaobject_list = object_repo.list(filters=ns_filters)
            object_list = [MetadefObject.to_wsme_model(db_metaobject, get_object_href(namespace, db_metaobject), self.obj_schema_link) for db_metaobject in db_metaobject_list]
            if object_list:
                namespace_detail.objects = object_list
            rs_repo = self.gateway.get_metadef_resource_type_repo(req.context)
            db_resource_type_list = rs_repo.list(filters=ns_filters)
            resource_type_list = [ResourceTypeAssociation.to_wsme_model(resource_type) for resource_type in db_resource_type_list]
            if resource_type_list:
                namespace_detail.resource_type_associations = resource_type_list
            prop_repo = self.gateway.get_metadef_property_repo(req.context)
            db_properties = prop_repo.list(filters=ns_filters)
            property_list = Namespace.to_model_properties(db_properties)
            if property_list:
                namespace_detail.properties = property_list
            if filters and filters['resource_type']:
                namespace_detail = self._prefix_property_name(namespace_detail, filters['resource_type'])
            tag_repo = self.gateway.get_metadef_tag_repo(req.context)
            db_metatag_list = tag_repo.list(filters=ns_filters)
            tag_list = [MetadefTag(**{'name': db_metatag.name}) for db_metatag in db_metatag_list]
            if tag_list:
                namespace_detail.tags = tag_list
        except exception.Forbidden as e:
            LOG.debug("User not permitted to show metadata namespace '%s'", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        return namespace_detail

    def update(self, req, user_ns, namespace):
        namespace_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            ns_obj = namespace_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=ns_obj, enforcer=self.policy).modify_metadef_namespace()
            ns_obj._old_namespace = ns_obj.namespace
            ns_obj.namespace = wsme_utils._get_value(user_ns.namespace)
            ns_obj.display_name = wsme_utils._get_value(user_ns.display_name)
            ns_obj.description = wsme_utils._get_value(user_ns.description)
            ns_obj.visibility = wsme_utils._get_value(user_ns.visibility) or 'private'
            ns_obj.protected = wsme_utils._get_value(user_ns.protected) or False
            ns_obj.owner = wsme_utils._get_value(user_ns.owner) or req.context.owner
            updated_namespace = namespace_repo.save(ns_obj)
        except exception.Invalid as e:
            msg = _("Couldn't update metadata namespace: %s") % encodeutils.exception_to_unicode(e)
            raise webob.exc.HTTPBadRequest(explanation=msg)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to update metadata namespace '%s'", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)
        except exception.Duplicate as e:
            raise webob.exc.HTTPConflict(explanation=e.msg)
        return Namespace.to_wsme_model(updated_namespace, get_namespace_href(updated_namespace), self.ns_schema_link)

    def delete(self, req, namespace):
        namespace_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = namespace_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).delete_metadef_namespace()
            namespace_obj.delete()
            namespace_repo.remove(namespace_obj)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata namespace '%s'", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def delete_objects(self, req, namespace):
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).delete_metadef_namespace()
            namespace_obj.delete()
            ns_repo.remove_objects(namespace_obj)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata objects within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def delete_tags(self, req, namespace):
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            policy_check = api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy)
            policy_check.delete_metadef_namespace()
            policy_check.delete_metadef_tags()
            namespace_obj.delete()
            ns_repo.remove_tags(namespace_obj)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata tags within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def delete_properties(self, req, namespace):
        ns_repo = self.gateway.get_metadef_namespace_repo(req.context)
        try:
            namespace_obj = ns_repo.get(namespace)
        except (exception.Forbidden, exception.NotFound):
            msg = _('Namespace %s not found') % namespace
            raise webob.exc.HTTPNotFound(explanation=msg)
        try:
            api_policy.MetadefAPIPolicy(req.context, md_resource=namespace_obj, enforcer=self.policy).delete_metadef_namespace()
            namespace_obj.delete()
            ns_repo.remove_properties(namespace_obj)
        except exception.Forbidden as e:
            LOG.debug("User not permitted to delete metadata properties within '%s' namespace", namespace)
            raise webob.exc.HTTPForbidden(explanation=e.msg)
        except exception.NotFound as e:
            raise webob.exc.HTTPNotFound(explanation=e.msg)

    def _prefix_property_name(self, namespace_detail, user_resource_type):
        prefix = None
        if user_resource_type and namespace_detail.resource_type_associations:
            for resource_type in namespace_detail.resource_type_associations:
                if resource_type.name == user_resource_type:
                    prefix = resource_type.prefix
                    break
        if prefix:
            if namespace_detail.properties:
                new_property_dict = dict()
                for key, value in namespace_detail.properties.items():
                    new_property_dict[prefix + key] = value
                namespace_detail.properties = new_property_dict
            if namespace_detail.objects:
                for object in namespace_detail.objects:
                    new_object_property_dict = dict()
                    for key, value in object.properties.items():
                        new_object_property_dict[prefix + key] = value
                    object.properties = new_object_property_dict
                    if object.required and len(object.required) > 0:
                        required = [prefix + name for name in object.required]
                        object.required = required
        return namespace_detail