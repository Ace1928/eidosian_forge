import six
from genshi.compat import numeric_types
from genshi.core import Attrs, Markup, Namespace, QName, Stream, \
class ElementFactory(object):
    """Factory for `Element` objects.
    
    A new element is created simply by accessing a correspondingly named
    attribute of the factory object:
    
    >>> factory = ElementFactory()
    >>> print(factory.foo)
    <foo/>
    >>> print(factory.foo(id=2))
    <foo id="2"/>
    
    Markup fragments (lists of nodes without a parent element) can be created
    by calling the factory:
    
    >>> print(factory('Hello, ', factory.em('world'), '!'))
    Hello, <em>world</em>!
    
    A factory can also be bound to a specific namespace:
    
    >>> factory = ElementFactory('http://www.w3.org/1999/xhtml')
    >>> print(factory.html(lang="en"))
    <html xmlns="http://www.w3.org/1999/xhtml" lang="en"/>
    
    The namespace for a specific element can be altered on an existing factory
    by specifying the new namespace using item access:
    
    >>> factory = ElementFactory()
    >>> print(factory.html(factory['http://www.w3.org/2000/svg'].g(id=3)))
    <html><g xmlns="http://www.w3.org/2000/svg" id="3"/></html>
    
    Usually, the `ElementFactory` class is not be used directly. Rather, the
    `tag` instance should be used to create elements.
    """

    def __init__(self, namespace=None):
        """Create the factory, optionally bound to the given namespace.
        
        :param namespace: the namespace URI for any created elements, or `None`
                          for no namespace
        """
        if namespace and (not isinstance(namespace, Namespace)):
            namespace = Namespace(namespace)
        self.namespace = namespace

    def __call__(self, *args):
        """Create a fragment that has the given positional arguments as child
        nodes.

        :return: the created `Fragment`
        :rtype: `Fragment`
        """
        return Fragment()(*args)

    def __getitem__(self, namespace):
        """Return a new factory that is bound to the specified namespace.
        
        :param namespace: the namespace URI or `Namespace` object
        :return: an `ElementFactory` that produces elements bound to the given
                 namespace
        :rtype: `ElementFactory`
        """
        return ElementFactory(namespace)

    def __getattr__(self, name):
        """Create an `Element` with the given name.
        
        :param name: the tag name of the element to create
        :return: an `Element` with the specified name
        :rtype: `Element`
        """
        return Element(self.namespace and self.namespace[name] or name)