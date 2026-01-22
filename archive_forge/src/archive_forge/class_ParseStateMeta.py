import logging
import re
from oslo_policy import _checks
class ParseStateMeta(type):
    """Metaclass for the :class:`.ParseState` class.

    Facilitates identifying reduction methods.
    """

    def __new__(mcs, name, bases, cls_dict):
        """Create the class.

        Injects the 'reducers' list, a list of tuples matching token sequences
        to the names of the corresponding reduction methods.
        """
        reducers = []
        for key, value in cls_dict.items():
            if not hasattr(value, 'reducers'):
                continue
            for reduction in value.reducers:
                reducers.append((reduction, key))
        cls_dict['reducers'] = reducers
        return super(ParseStateMeta, mcs).__new__(mcs, name, bases, cls_dict)