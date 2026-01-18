import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError

        Validate the new length for a proposed operation.

        Parameters
        ----------
        new_length : int
            New length of the list.

        Raises
        ------
        TraitError
            If the proposed new length would violate the length constraints
            of the list.
        