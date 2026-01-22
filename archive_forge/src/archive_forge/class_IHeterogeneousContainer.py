from pyomo.core.kernel.base import (
class IHeterogeneousContainer(ICategorizedObjectContainer):
    """
    A partial implementation of the ICategorizedObjectContainer
    interface for implementations that store multiple
    categories of objects.

    Complete implementations need to set the _ctype
    attribute and declare the remaining required abstract
    properties of the ICategorizedObjectContainer base
    class.
    """
    __slots__ = ()
    _is_heterogeneous_container = True

    def collect_ctypes(self, active=True, descend_into=True):
        """Returns the set of object category types that can
        be found under this container.

        Args:
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            A set of object category types
        """
        assert active in (None, True)
        ctypes = set()
        if active is not None and (not self.active):
            return ctypes
        descend_into = _convert_descend_into(descend_into)
        for child_ctype in self.child_ctypes():
            for obj in self.components(ctype=child_ctype, active=active, descend_into=_convert_descend_into._false):
                ctypes.add(child_ctype)
                break
        if descend_into is _convert_descend_into._false:
            return ctypes
        for child_ctype in tuple(ctypes):
            if child_ctype._is_heterogeneous_container:
                for obj in self.components(ctype=child_ctype, active=active, descend_into=_convert_descend_into._false):
                    assert obj._is_heterogeneous_container
                    if descend_into(obj):
                        ctypes.update(obj.collect_ctypes(active=active, descend_into=descend_into))
        return ctypes

    def child_ctypes(self, *args, **kwds):
        """Returns the set of child object category types
        stored in this container."""
        raise NotImplementedError

    def components(self, ctype=_no_ctype, active=True, descend_into=True):
        """
        Generates an efficient traversal of all components
        stored under this container. Components are
        categorized objects that are either (1) not
        containers, or (2) are heterogeneous containers.

        Args:
            ctype: Indicates the category of components to
                include. The default value indicates that
                all categories should be included.
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            iterator of components in the storage tree
        """
        assert active in (None, True)
        if active is not None and (not self.active):
            return
        ctype = _convert_ctype.get(ctype, ctype)
        descend_into = _convert_descend_into(descend_into)
        if ctype is _no_ctype:
            for child in self.children():
                if active is not None and (not child.active):
                    continue
                if not child._is_container:
                    yield child
                elif child._is_heterogeneous_container:
                    yield child
                    if descend_into(child):
                        yield from child.components(active=active, descend_into=descend_into)
                elif descend_into is _convert_descend_into._false or not child.ctype._is_heterogeneous_container:
                    assert child._is_container
                    yield from child.components(active=active)
                else:
                    assert child._is_container
                    for obj in child.components(active=active):
                        assert obj._is_heterogeneous_container
                        yield obj
                        if descend_into(obj):
                            yield from obj.components(active=active, descend_into=descend_into)
        else:
            for item in heterogeneous_containers(self, active=active, descend_into=descend_into):
                for child in item.children(ctype=ctype):
                    if not child._is_container or child._is_heterogeneous_container:
                        if active is None or child.active:
                            yield child
                    else:
                        assert child._is_container
                        yield from child.components(active=active)