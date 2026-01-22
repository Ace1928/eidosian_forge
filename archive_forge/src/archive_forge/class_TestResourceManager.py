import heapq
import inspect
import unittest
from pbr.version import VersionInfo
class TestResourceManager(object):
    """A manager for resources that can be shared across tests.

    ResourceManagers can report activity to a TestResult. The methods
     - startCleanResource(resource)
     - stopCleanResource(resource)
     - startMakeResource(resource)
     - stopMakeResource(resource)
    will be looked for and if present invoked before and after cleaning or
    creation of resource objects takes place.

    :cvar resources: The same as the resources list on an instance, the default
        constructor will look for the class instance and copy it. This is a
        convenience to avoid needing to define __init__ solely to alter the
        dependencies list.
    :ivar resources: The resources that this resource needs. Calling
        neededResources will return the closure of this resource and its needed
        resources. The resources list is in the same format as resources on a
        test case - a list of tuples (attribute_name, resource).
    :ivar setUpCost: The relative cost to construct a resource of this type.
         One good approach is to set this to the number of seconds it normally
         takes to set up the resource.
    :ivar tearDownCost: The relative cost to tear down a resource of this
         type. One good approach is to set this to the number of seconds it
         normally takes to tear down the resource.
    """
    setUpCost = 1
    tearDownCost = 1

    def __init__(self):
        """Create a TestResourceManager object."""
        self._dirty = False
        self._uses = 0
        self._currentResource = None
        self.resources = list(getattr(self.__class__, 'resources', []))

    def _call_result_method_if_exists(self, result, methodname, *args):
        """Call a method on a TestResult that may exist."""
        method = getattr(result, methodname, None)
        if callable(method):
            method(*args)

    def _clean_all(self, resource, result):
        """Clean the dependencies from resource, and then resource itself."""
        self._call_result_method_if_exists(result, 'startCleanResource', self)
        self.clean(resource)
        for name, manager in self.resources:
            manager.finishedWith(getattr(resource, name))
        self._call_result_method_if_exists(result, 'stopCleanResource', self)

    def clean(self, resource):
        """Override this to class method to hook into resource removal."""

    def dirtied(self, resource):
        """Mark the resource as having been 'dirtied'.

        A resource is dirty when it is no longer suitable for use by other
        tests.

        e.g. a shared database that has had rows changed.
        """
        self._dirty = True

    def finishedWith(self, resource, result=None):
        """Indicate that 'resource' has one less user.

        If there are no more registered users of 'resource' then we trigger
        the `clean` hook, which should do any resource-specific
        cleanup.

        :param resource: A resource returned by
            `TestResourceManager.getResource`.
        :param result: An optional TestResult to report resource changes to.
        """
        self._uses -= 1
        if self._uses == 0:
            self._clean_all(resource, result)
            self._setResource(None)

    def getResource(self, result=None):
        """Get the resource for this class and record that it's being used.

        The resource is constructed using the `make` hook.

        Once done with the resource, pass it to `finishedWith` to indicated
        that it is no longer needed.
        :param result: An optional TestResult to report resource changes to.
        """
        if self._uses == 0:
            self._setResource(self._make_all(result))
        elif self.isDirty():
            self._setResource(self.reset(self._currentResource, result))
        self._uses += 1
        return self._currentResource

    def isDirty(self):
        """Return True if this managers cached resource is dirty.

        Calling when the resource is not currently held has undefined
        behaviour.
        """
        if self._dirty:
            return True
        for name, mgr in self.resources:
            if mgr.isDirty():
                return True
            res = mgr.getResource()
            try:
                if res is not getattr(self._currentResource, name):
                    return True
            finally:
                mgr.finishedWith(res)

    def _make_all(self, result):
        """Make the dependencies of this resource and this resource."""
        self._call_result_method_if_exists(result, 'startMakeResource', self)
        dependency_resources = {}
        for name, resource in self.resources:
            dependency_resources[name] = resource.getResource()
        resource = self.make(dependency_resources)
        for name, value in dependency_resources.items():
            setattr(resource, name, value)
        self._call_result_method_if_exists(result, 'stopMakeResource', self)
        return resource

    def make(self, dependency_resources):
        """Override this to construct resources.

        :param dependency_resources: A dict mapping name -> resource instance
            for the resources specified as dependencies.
        :return: The made resource.
        """
        raise NotImplementedError('Override make to construct resources.')

    def neededResources(self):
        """Return the resources needed for this resource, including self.

        :return: A list of needed resources, in topological deepest-first
            order.
        """
        seen = set([self])
        result = []
        for name, resource in self.resources:
            for resource in resource.neededResources():
                if resource in seen:
                    continue
                seen.add(resource)
                result.append(resource)
        result.append(self)
        return result

    def reset(self, old_resource, result=None):
        """Return a clean version of old_resource.

        By default, the resource will be cleaned then remade if it had
        previously been `dirtied` by the helper self._reset() - which is the
        extension point folk should override to customise reset behaviour.

        This function takes the dependent resource stack into consideration as
        _make_all and _clean_all do. The inconsistent naming is because reset
        is part of the public interface, but _make_all and _clean_all is not.

        Note that if a resource A holds a lock or other blocking thing on
        a dependency D, reset will result in this call sequence over a 
        getResource(), dirty(), getResource(), finishedWith(), finishedWith()
        sequence:
        B.make(), A.make(), B.reset(), A.reset(), A.clean(), B.clean()
        Thus it is important that B.reset not assume that A has been cleaned or
        reset before B is reset: it should arrange to reference count, lazy
        cleanup or forcibly reset resource in some fashion.

        As an example, consider that B is a database with sample data, and
        A is an application server serving content from it. B._reset() should
        disconnect all database clients, reset the state of the database, and
        A._reset() should tell the application server to dump any internal
        caches it might have.

        In principle we might make a richer API to allow before-and-after
        reset actions, but so far that hasn't been needed.

        :return: The possibly new resource.
        :param result: An optional TestResult to report resource changes to.
        """
        if not self.isDirty():
            return old_resource
        self._call_result_method_if_exists(result, 'startResetResource', self)
        dependency_resources = {}
        for name, mgr in self.resources:
            dependency_resources[name] = mgr.reset(getattr(old_resource, name), result)
        resource = self._reset(old_resource, dependency_resources)
        for name, value in dependency_resources.items():
            setattr(resource, name, value)
        self._call_result_method_if_exists(result, 'stopResetResource', self)
        return resource

    def _reset(self, resource, dependency_resources):
        """Override this to reset resources other than via clean+make.

        This method should reset the self._dirty flag (assuming the manager can
        ever be clean) and return either the old resource cleaned or a fresh
        one.

        :param resource: The resource to reset.
        :param dependency_resources: A dict mapping name -> resource instance
            for the resources specified as dependencies.
        """
        self.clean(resource)
        return self.make(dependency_resources)

    def _setResource(self, new_resource):
        """Set the current resource to a new value."""
        self._currentResource = new_resource
        self._dirty = False