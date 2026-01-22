from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
class InterfaceChecker(HasTraits):
    """ Checks that interfaces are actually implemented.
    """

    def check_implements(self, cls, interfaces, error_mode):
        """ Checks that the class implements the specified interfaces.

            'interfaces' can be a single interface or a list of interfaces.
        """
        try:
            iter(interfaces)
        except TypeError:
            interfaces = [interfaces]
        if issubclass(cls, HasTraits):
            for interface in interfaces:
                if not self._check_has_traits_class(cls, interface, error_mode):
                    return False
        else:
            for interface in interfaces:
                if not self._check_non_has_traits_class(cls, interface, error_mode):
                    return False
        return True

    def _check_has_traits_class(self, cls, interface, error_mode):
        """ Checks that a 'HasTraits' class implements an interface.
        """
        return self._check_traits(cls, interface, error_mode) and self._check_methods(cls, interface, error_mode)

    def _check_non_has_traits_class(self, cls, interface, error_mode):
        """ Checks that a non-'HasTraits' class implements an interface.
        """
        return self._check_methods(cls, interface, error_mode)

    def _check_methods(self, cls, interface, error_mode):
        """ Checks that a class implements the methods on an interface.
        """
        cls_methods = self._get_public_methods(cls)
        interface_methods = self._get_public_methods(interface)
        for name in interface_methods:
            if name not in cls_methods:
                return self._handle_error(MISSING_METHOD % (self._class_name(cls), name, self._class_name(interface)), error_mode)
            cls_argspec = getfullargspec(cls_methods[name])
            interface_argspec = getfullargspec(interface_methods[name])
            if cls_argspec != interface_argspec:
                return self._handle_error(BAD_SIGNATURE % (self._class_name(cls), name, self._class_name(interface)), error_mode)
        return True

    def _check_traits(self, cls, interface, error_mode):
        """ Checks that a class implements the traits on an interface.
        """
        missing = set(interface.class_traits()).difference(set(cls.class_traits()))
        if len(missing) > 0:
            return self._handle_error(MISSING_TRAIT % (self._class_name(cls), repr(list(missing))[1:-1], self._class_name(interface)), error_mode)
        return True

    def _get_public_methods(self, cls):
        """ Returns all public methods on a class.

            Returns a dictionary containing all public methods keyed by name.
        """
        public_methods = {}
        for c in getmro(cls):
            if c is HasTraits:
                break
            for name, value in c.__dict__.items():
                if not name.startswith('_') and type(value) is FunctionType:
                    if name not in public_methods:
                        public_methods[name] = value
        return public_methods

    def _class_name(self, cls):
        return cls.__name__

    def _handle_error(self, msg, error_mode):
        if error_mode > 1:
            raise InterfaceError(msg)
        if error_mode == 1:
            logger.warning(msg)
        return False