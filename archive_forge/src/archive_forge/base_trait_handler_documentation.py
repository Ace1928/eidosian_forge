from .constants import DefaultValue
from .trait_errors import TraitError
 Returns a tuple containing the *inner traits* for this trait. Most
            trait handlers do not have any inner traits, and so will return an
            empty tuple. The exceptions are **List** and **Dict** trait types,
            which have inner traits used to validate the values assigned to the
            trait. For example, in *List( Int )*, the *inner traits* for
            **List** are ( **Int**, ).
        