from .base_trait_handler import BaseTraitHandler
from .trait_base import class_of
from .trait_errors import TraitError
 Verifies whether a new value assigned to a trait attribute is
        valid.

        This method *must* be implemented by subclasses of TraitHandler. It is
        called whenever a new value is assigned to a trait attribute defined
        using this trait handler.

        If the value received by validate() is not valid for the trait
        attribute, the method must called the predefined error() method to
        raise a TraitError exception

        Parameters
        ----------
        object : HasTraits instance
            The object whose attribute is being assigned.
        name : str
            The name of the attribute being assigned.
        value : any
            The proposed new value for the attribute.

        Returns
        -------
        any
            If the new value is valid, this method must return either the
            original value passed to it, or an alternate value to be assigned
            in place of the original value. Whatever value this method returns
            is the actual value assigned to *object.name*.

        