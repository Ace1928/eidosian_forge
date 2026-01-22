import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
class _PolicyBase:
    """Policy Object basic framework.

    This class is useless unless subclassed.  A subclass should define
    class attributes with defaults for any values that are to be
    managed by the Policy object.  The constructor will then allow
    non-default values to be set for these attributes at instance
    creation time.  The instance will be callable, taking these same
    attributes keyword arguments, and returning a new instance
    identical to the called instance except for those values changed
    by the keyword arguments.  Instances may be added, yielding new
    instances with any non-default values from the right hand
    operand overriding those in the left hand operand.  That is,

        A + B == A(<non-default values of B>)

    The repr of an instance can be used to reconstruct the object
    if and only if the repr of the values can be used to reconstruct
    those values.

    """

    def __init__(self, **kw):
        """Create new Policy, possibly overriding some defaults.

        See class docstring for a list of overridable attributes.

        """
        for name, value in kw.items():
            if hasattr(self, name):
                super(_PolicyBase, self).__setattr__(name, value)
            else:
                raise TypeError('{!r} is an invalid keyword argument for {}'.format(name, self.__class__.__name__))

    def __repr__(self):
        args = ['{}={!r}'.format(name, value) for name, value in self.__dict__.items()]
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))

    def clone(self, **kw):
        """Return a new instance with specified attributes changed.

        The new instance has the same attribute values as the current object,
        except for the changes passed in as keyword arguments.

        """
        newpolicy = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            object.__setattr__(newpolicy, attr, value)
        for attr, value in kw.items():
            if not hasattr(self, attr):
                raise TypeError('{!r} is an invalid keyword argument for {}'.format(attr, self.__class__.__name__))
            object.__setattr__(newpolicy, attr, value)
        return newpolicy

    def __setattr__(self, name, value):
        if hasattr(self, name):
            msg = '{!r} object attribute {!r} is read-only'
        else:
            msg = '{!r} object has no attribute {!r}'
        raise AttributeError(msg.format(self.__class__.__name__, name))

    def __add__(self, other):
        """Non-default values from right operand override those from left.

        The object returned is a new instance of the subclass.

        """
        return self.clone(**other.__dict__)