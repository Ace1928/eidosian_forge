from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Generic, TypeVar
class PropertyDescriptorFactory(Generic[T]):
    """ Base class for all Bokeh properties.

    A Bokeh property really consist of two parts: the familiar "property"
    portion, such as ``Int``, ``String``, etc., as well as an associated
    Python descriptor that delegates attribute access (e.g. ``range.start``)
    to the property instance.

    Consider the following class definition:

    .. code-block:: python

        from bokeh.model import Model
        from bokeh.core.properties import Int

        class SomeModel(Model):
            foo = Int(default=10)

    Then we can observe the following:

    .. code-block:: python

        >>> m = SomeModel()

        # The class itself has had a descriptor for 'foo' installed
        >>> getattr(SomeModel, 'foo')
        <bokeh.core.property.descriptors.PropertyDescriptor at 0x1065ffb38>

        # which is used when 'foo' is accessed on instances
        >>> m.foo
        10

    """

    def make_descriptors(self, name: str) -> list[PropertyDescriptor[T]]:
        """ Return a list of ``PropertyDescriptor`` instances to install on a
        class, in order to delegate attribute access to this property.

        Args:
            name (str) : the name of the property these descriptors are for

        Returns:
            list[PropertyDescriptor]

        The descriptors returned are collected by the ``MetaHasProps``
        metaclass and added to ``HasProps`` subclasses during class creation.

        Subclasses of ``PropertyDescriptorFactory`` are responsible for
        implementing this function to return descriptors specific to their
        needs.

        """
        raise NotImplementedError('make_descriptors not implemented')