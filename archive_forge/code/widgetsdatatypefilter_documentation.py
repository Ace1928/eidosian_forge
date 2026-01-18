import os
from warnings import warn
from nbconvert.utils.base import NbConvertBase
Return the first available format in the priority.

        Produces a UserWarning if no compatible mimetype is found.

        `output` is dict with structure {mimetype-of-element: value-of-element}

        