import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
class ResourceRegistry(object):
    """By looking at the environment, find the resource implementation."""

    def __init__(self, global_registry, param_defaults):
        self._registry = {'resources': {}}
        self.global_registry = global_registry
        self.param_defaults = param_defaults

    def load(self, json_snippet):
        self._load_registry([], json_snippet)

    def register_class(self, resource_type, resource_class, path=None):
        if path is None:
            path = [resource_type]
        ri = ResourceInfo(self, path, resource_class)
        self._register_info(path, ri)

    def _load_registry(self, path, registry):
        for k, v in iter(registry.items()):
            if v is None:
                self._register_info(path + [k], None)
            elif is_hook_definition(k, v) or is_valid_restricted_action(k, v):
                self._register_item(path + [k], v)
            elif isinstance(v, dict):
                self._load_registry(path + [k], v)
            else:
                self._register_info(path + [k], ResourceInfo(self, path + [k], v))

    def _register_item(self, path, item):
        name = path[-1]
        registry = self._registry
        for key in path[:-1]:
            if key not in registry:
                registry[key] = {}
            registry = registry[key]
        registry[name] = item

    def _register_info(self, path, info):
        """Place the new info in the correct location in the registry.

        :param path: a list of keys ['resources', 'my_srv', 'OS::Nova::Server']
        """
        descriptive_path = '/'.join(path)
        name = path[-1]
        registry = self._registry
        for key in path[:-1]:
            if key not in registry:
                registry[key] = {}
            registry = registry[key]
        if info is None:
            if name.endswith('*'):
                for res_name, reg_info in list(registry.items()):
                    if isinstance(reg_info, ResourceInfo) and res_name.startswith(name[:-1]):
                        LOG.warning('Removing %(item)s from %(path)s', {'item': res_name, 'path': descriptive_path})
                        del registry[res_name]
            else:
                LOG.warning('Removing %(item)s from %(path)s', {'item': name, 'path': descriptive_path})
                registry.pop(name, None)
            return
        if name in registry and isinstance(registry[name], ResourceInfo):
            if registry[name] == info:
                return
            details = {'path': descriptive_path, 'was': str(registry[name].value), 'now': str(info.value)}
            LOG.warning('Changing %(path)s from %(was)s to %(now)s', details)
        if isinstance(info, ClassResourceInfo):
            if info.value.support_status.status != support.SUPPORTED:
                if info.value.support_status.message is not None:
                    details = {'name': info.name, 'status': str(info.value.support_status.status), 'message': str(info.value.support_status.message)}
                    LOG.warning('%(name)s is %(status)s. %(message)s', details)
        info.user_resource = self.global_registry is not None
        registry[name] = info

    def log_resource_info(self, show_all=False, prefix=None):
        registry = self._registry
        prefix = '%s ' % prefix if prefix is not None else ''
        for name in registry:
            if name == 'resources':
                continue
            if show_all or isinstance(registry[name], TemplateResourceInfo):
                msg = '%(p)sRegistered: %(t)s' % {'p': prefix, 't': str(registry[name])}
                LOG.info(msg)

    def remove_item(self, info):
        if not isinstance(info, TemplateResourceInfo):
            return
        registry = self._registry
        for key in info.path[:-1]:
            registry = registry[key]
        if info.path[-1] in registry:
            registry.pop(info.path[-1])

    def get_rsrc_restricted_actions(self, resource_name):
        """Returns a set of restricted actions.

        For a given resource we get the set of restricted actions.

        Actions are set in this format via `resources`::

            {
                "restricted_actions": [update, replace]
            }

        A restricted_actions value is either `update`, `replace` or a list
        of those values. Resources support wildcard matching. The asterisk
        sign matches everything.
        """
        ress = self._registry['resources']
        restricted_actions = set()
        for name_pattern, resource in ress.items():
            if fnmatch.fnmatchcase(resource_name, name_pattern):
                if 'restricted_actions' in resource:
                    actions = resource['restricted_actions']
                    if isinstance(actions, str):
                        restricted_actions.add(actions)
                    elif isinstance(actions, collections.abc.Sequence):
                        restricted_actions |= set(actions)
        return restricted_actions

    def matches_hook(self, resource_name, hook):
        """Return whether a resource have a hook set in the environment.

        For a given resource and a hook type, we check to see if the passed
        group of resources has the right hook associated with the name.

        Hooks are set in this format via `resources`::

            {
                "res_name": {
                    "hooks": [pre-create, pre-update]
                },
                "*_suffix": {
                    "hooks": pre-create
                },
                "prefix_*": {
                    "hooks": pre-update
                }
            }

        A hook value is either `pre-create`, `pre-update` or a list of those
        values. Resources support wildcard matching. The asterisk sign matches
        everything.
        """
        ress = self._registry['resources']
        for name_pattern, resource in ress.items():
            if fnmatch.fnmatchcase(resource_name, name_pattern):
                if 'hooks' in resource:
                    hooks = resource['hooks']
                    if isinstance(hooks, str):
                        if hook == hooks:
                            return True
                    elif isinstance(hooks, collections.abc.Sequence):
                        if hook in hooks:
                            return True
        return False

    def remove_resources_except(self, resource_name):
        ress = self._registry['resources']
        new_resources = {}
        for name, res in ress.items():
            if fnmatch.fnmatchcase(resource_name, name):
                new_resources.update(res)
        if resource_name in ress:
            new_resources.update(ress[resource_name])
        self._registry['resources'] = new_resources

    def iterable_by(self, resource_type, resource_name=None):
        is_templ_type = resource_type.endswith(('.yaml', '.template'))
        if self.global_registry is not None and is_templ_type:
            if resource_type not in self._registry:
                res = ResourceInfo(self, [resource_type], None)
                self._register_info([resource_type], res)
            yield self._registry[resource_type]
        if resource_name:
            impl = self._registry['resources'].get(resource_name)
            if impl and resource_type in impl:
                yield impl[resource_type]
        impl = self._registry.get(resource_type)
        if impl:
            yield impl

        def is_a_glob(resource_type):
            return resource_type.endswith('*')
        globs = filter(is_a_glob, iter(self._registry))
        for pattern in globs:
            if self._registry[pattern].matches(resource_type):
                yield self._registry[pattern]

    def get_resource_info(self, resource_type, resource_name=None, registry_type=None, ignore=None):
        """Find possible matches to the resource type and name.

        Chain the results from the global and user registry to find
        a match.
        """
        if self.global_registry is not None:
            giter = self.global_registry.iterable_by(resource_type, resource_name)
        else:
            giter = []
        matches = itertools.chain(self.iterable_by(resource_type, resource_name), giter)
        for info in sorted(matches):
            try:
                match = info.get_resource_info(resource_type, resource_name)
            except exception.EntityNotFound:
                continue
            if registry_type is None or isinstance(match, registry_type):
                if ignore is not None and match == ignore:
                    continue
                if match and (not match.user_resource) and (not isinstance(info, (TemplateResourceInfo, ClassResourceInfo))):
                    self._register_info([resource_type], info)
                return match
        raise exception.EntityNotFound(entity='Resource Type', name=resource_type)

    def get_class(self, resource_type, resource_name=None, files=None):
        info = self.get_resource_info(resource_type, resource_name=resource_name)
        return info.get_class(files=files)

    def get_class_to_instantiate(self, resource_type, resource_name=None):
        if resource_type == '':
            msg = _('Resource "%s" has no type') % resource_name
            raise exception.StackValidationFailed(message=msg)
        elif resource_type is None:
            msg = _('Non-empty resource type is required for resource "%s"') % resource_name
            raise exception.StackValidationFailed(message=msg)
        elif not isinstance(resource_type, str):
            msg = _('Resource "%s" type is not a string') % resource_name
            raise exception.StackValidationFailed(message=msg)
        try:
            info = self.get_resource_info(resource_type, resource_name=resource_name)
        except exception.EntityNotFound as exc:
            raise exception.StackValidationFailed(message=str(exc))
        return info.get_class_to_instantiate()

    def as_dict(self):
        """Return user resources in a dict format."""

        def _as_dict(level):
            tmp = {}
            for k, v in iter(level.items()):
                if isinstance(v, dict):
                    tmp[k] = _as_dict(v)
                elif is_hook_definition(k, v) or is_valid_restricted_action(k, v):
                    tmp[k] = v
                elif v.user_resource:
                    tmp[k] = v.value
            return tmp
        return _as_dict(self._registry)

    def get_types(self, cnxt=None, support_status=None, type_name=None, version=None, with_description=False):
        """Return a list of valid resource types."""
        if support_status is not None and (not support.is_valid_status(support_status)):
            msg = _('Invalid support status and should be one of %s') % str(support.SUPPORT_STATUSES)
            raise exception.Invalid(reason=msg)
        enforcer = policy.ResourceEnforcer()
        if type_name is not None:
            try:
                name_exp = re.compile(type_name)
            except Exception:
                return []
        else:
            name_exp = None

        def matches(name, info):
            if not isinstance(info, (ClassResourceInfo, TemplateResourceInfo)):
                return False
            if name_exp is not None and (not name_exp.match(name)):
                return False
            rsrc_cls = info.get_class_to_instantiate()
            if rsrc_cls.support_status.status == support.HIDDEN:
                return False
            if version is not None and rsrc_cls.support_status.version != version:
                return False
            if support_status is not None and rsrc_cls.support_status.status != support_status:
                return False
            if cnxt is not None:
                try:
                    enforcer.enforce(cnxt, name, is_registered_policy=True)
                except enforcer.exc:
                    return False
                try:
                    avail, err = rsrc_cls.is_service_available(cnxt)
                except Exception:
                    avail = False
                if not avail:
                    return False
            return True
        import heat.engine.resource

        def resource_description(name, info):
            if not with_description:
                return name
            rsrc_cls = info.get_class()
            if rsrc_cls is None:
                rsrc_cls = heat.engine.resource.Resource
            return {'resource_type': name, 'description': rsrc_cls.getdoc()}
        return [resource_description(name, info) for name, info in self._registry.items() if matches(name, info)]