from functools import partial
from .constants import DefaultValue
 Returns the 'mapped trait' definition for a mapped trait, the default
        value of which is a callable that maps the value of the original trait.

        Parameters
        ----------
        trait : ctrait.CTrait
            A trait for which the 'mapped trait' definition is being created.
        name : str
            The name of the trait for which the 'mapped trait' definition is
            being created.

        Returns
        -------
        trait_types.Any
            A definition of the 'mapped trait'
    