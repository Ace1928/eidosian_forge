import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
class RestrictionType(type):
    """RestrictionType. Type from which all enzyme classes are derived.

    Implement the operator methods.
    """

    def __init__(cls, name='', bases=(), dct=None):
        """Initialize RestrictionType instance.

        Not intended to be used in normal operation. The enzymes are
        instantiated when importing the module.
        See below.
        """
        if '-' in name:
            raise ValueError(f'Problem with hyphen in {name!r} as enzyme name')
        try:
            cls.compsite = re.compile(cls.compsite)
        except AttributeError:
            pass
        except Exception:
            raise ValueError(f'Problem with regular expression, re.compiled({cls.compsite!r})') from None

    def __add__(cls, other):
        """Add restriction enzyme to a RestrictionBatch().

        If other is an enzyme returns a batch of the two enzymes.
        If other is already a RestrictionBatch add enzyme to it.
        """
        if isinstance(other, RestrictionType):
            return RestrictionBatch([cls, other])
        elif isinstance(other, RestrictionBatch):
            return other.add_nocheck(cls)
        else:
            raise TypeError

    def __truediv__(cls, other):
        """Override '/' operator to use as search method.

        >>> from Bio.Restriction import EcoRI
        >>> EcoRI/Seq('GAATTC')
        [2]

        Returns RE.search(other).
        """
        return cls.search(other)

    def __rtruediv__(cls, other):
        """Override division with reversed operands to use as search method.

        >>> from Bio.Restriction import EcoRI
        >>> Seq('GAATTC')/EcoRI
        [2]

        Returns RE.search(other).
        """
        return cls.search(other)

    def __floordiv__(cls, other):
        """Override '//' operator to use as catalyse method.

        >>> from Bio.Restriction import EcoRI
        >>> EcoRI//Seq('GAATTC')
        (Seq('G'), Seq('AATTC'))

        Returns RE.catalyse(other).
        """
        return cls.catalyse(other)

    def __rfloordiv__(cls, other):
        """As __floordiv__, with reversed operands.

        >>> from Bio.Restriction import EcoRI
        >>> Seq('GAATTC')//EcoRI
        (Seq('G'), Seq('AATTC'))

        Returns RE.catalyse(other).
        """
        return cls.catalyse(other)

    def __str__(cls):
        """Return the name of the enzyme as string."""
        return cls.__name__

    def __repr__(cls):
        """Implement repr method.

        Used with eval or exec will instantiate the enzyme.
        """
        return f'{cls.__name__}'

    def __len__(cls):
        """Return length of recognition site of enzyme as int."""
        try:
            return cls.size
        except AttributeError:
            return 0

    def __hash__(cls):
        """Implement ``hash()`` method for ``RestrictionType``.

        Python default is to use ``id(...)``
        This is consistent with the ``__eq__`` implementation
        """
        return id(cls)

    def __eq__(cls, other):
        """Override '==' operator.

        True if RE and other are the same enzyme.

        Specifically this checks they are the same Python object.
        """
        return id(cls) == id(other)

    def __ne__(cls, other):
        """Override '!=' operator.

        Isoschizomer strict (same recognition site, same restriction) -> False
        All the other-> True

        WARNING - This is not the inverse of the __eq__ method

        >>> from Bio.Restriction import SacI, SstI
        >>> SacI != SstI  # true isoschizomers
        False
        >>> SacI == SstI
        False
        """
        if not isinstance(other, RestrictionType):
            return True
        elif cls.charac == other.charac:
            return False
        else:
            return True

    def __rshift__(cls, other):
        """Override '>>' operator to test for neoschizomers.

        neoschizomer : same recognition site, different restriction. -> True
        all the others :                                             -> False

        >>> from Bio.Restriction import SmaI, XmaI
        >>> SmaI >> XmaI
        True
        """
        if not isinstance(other, RestrictionType):
            return False
        elif cls.site == other.site and cls.charac != other.charac:
            return True
        else:
            return False

    def __mod__(cls, other):
        """Override '%' operator to test for compatible overhangs.

        True if a and b have compatible overhang.

        >>> from Bio.Restriction import XhoI, SalI
        >>> XhoI % SalI
        True
        """
        if not isinstance(other, RestrictionType):
            raise TypeError(f'expected RestrictionType, got {type(other)} instead')
        return cls._mod1(other)

    def __ge__(cls, other):
        """Compare length of recognition site of two enzymes.

        Override '>='. a is greater or equal than b if the a site is longer
        than b site. If their site have the same length sort by alphabetical
        order of their names.

        >>> from Bio.Restriction import EcoRI, EcoRV
        >>> EcoRI.size
        6
        >>> EcoRV.size
        6
        >>> EcoRI >= EcoRV
        False
        """
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        if len(cls) > len(other):
            return True
        elif cls.size == len(other) and cls.__name__ >= other.__name__:
            return True
        else:
            return False

    def __gt__(cls, other):
        """Compare length of recognition site of two enzymes.

        Override '>'. Sorting order:

        1. size of the recognition site.
        2. if equal size, alphabetical order of the names.

        """
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        if len(cls) > len(other):
            return True
        elif cls.size == len(other) and cls.__name__ > other.__name__:
            return True
        else:
            return False

    def __le__(cls, other):
        """Compare length of recognition site of two enzymes.

        Override '<='. Sorting order:

        1. size of the recognition site.
        2. if equal size, alphabetical order of the names.

        """
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        elif len(cls) < len(other):
            return True
        elif len(cls) == len(other) and cls.__name__ <= other.__name__:
            return True
        else:
            return False

    def __lt__(cls, other):
        """Compare length of recognition site of two enzymes.

        Override '<'. Sorting order:

        1. size of the recognition site.
        2. if equal size, alphabetical order of the names.

        """
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        elif len(cls) < len(other):
            return True
        elif len(cls) == len(other) and cls.__name__ < other.__name__:
            return True
        else:
            return False