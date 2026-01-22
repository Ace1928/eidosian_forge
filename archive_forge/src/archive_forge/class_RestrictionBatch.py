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
class RestrictionBatch(set):
    """Class for operations on more than one enzyme."""

    def __init__(self, first=(), suppliers=()):
        """Initialize empty RB or pre-fill with enzymes (from supplier)."""
        first = [self.format(x) for x in first]
        first += [eval(x) for n in suppliers for x in suppliers_dict[n][1]]
        set.__init__(self, first)
        self.mapping = dict.fromkeys(self)
        self.already_mapped = None
        self.suppliers = [x for x in suppliers if x in suppliers_dict]

    def __str__(self):
        """Return a readable representation of the ``RestrictionBatch``."""
        if len(self) < 5:
            return '+'.join(self.elements())
        else:
            return '...'.join(('+'.join(self.elements()[:2]), '+'.join(self.elements()[-2:])))

    def __repr__(self):
        """Represent ``RestrictionBatch`` class as a string for debugging."""
        return f'RestrictionBatch({self.elements()})'

    def __contains__(self, other):
        """Implement ``in`` for ``RestrictionBatch``."""
        try:
            other = self.format(other)
        except ValueError:
            return False
        return set.__contains__(self, other)

    def __div__(self, other):
        """Override '/' operator to use as search method."""
        return self.search(other)

    def __rdiv__(self, other):
        """Override division with reversed operands to use as search method."""
        return self.search(other)

    def __truediv__(self, other):
        """Override Python 3 division operator to use as search method.

        Like __div__.
        """
        return self.search(other)

    def __rtruediv__(self, other):
        """As __truediv___, with reversed operands.

        Like __rdiv__.
        """
        return self.search(other)

    def get(self, enzyme, add=False):
        """Check if enzyme is in batch and return it.

        If add is True and enzyme is not in batch add enzyme to batch.
        If add is False (which is the default) only return enzyme.
        If enzyme is not a RestrictionType or can not be evaluated to
        a RestrictionType, raise a ValueError.
        """
        e = self.format(enzyme)
        if e in self:
            return e
        elif add:
            self.add(e)
            return e
        else:
            raise ValueError(f'enzyme {e.__name__} is not in RestrictionBatch')

    def lambdasplit(self, func):
        """Filter enzymes in batch with supplied function.

        The new batch will contain only the enzymes for which
        func return True.
        """
        d = list(filter(func, self))
        new = RestrictionBatch()
        new._data = dict(zip(d, [True] * len(d)))
        return new

    def add_supplier(self, letter):
        """Add all enzymes from a given supplier to batch.

        letter represents the suppliers as defined in the dictionary
        RestrictionDictionary.suppliers
        Returns None.
        Raise a KeyError if letter is not a supplier code.
        """
        supplier = suppliers_dict[letter]
        self.suppliers.append(letter)
        for x in supplier[1]:
            self.add_nocheck(eval(x))

    def current_suppliers(self):
        """List the current suppliers for the restriction batch.

        Return a sorted list of the suppliers which have been used to
        create the batch.
        """
        suppl_list = sorted((suppliers_dict[x][0] for x in self.suppliers))
        return suppl_list

    def __iadd__(self, other):
        """Override '+=' for use with sets.

        b += other -> add other to b, check the type of other.
        """
        self.add(other)
        return self

    def __add__(self, other):
        """Override '+' for use with sets.

        b + other -> new RestrictionBatch.
        """
        new = self.__class__(self)
        new.add(other)
        return new

    def remove(self, other):
        """Remove enzyme from restriction batch.

        Safe set.remove method. Verify that other is a RestrictionType or can
        be evaluated to a RestrictionType.
        Raise a ValueError if other can not be evaluated to a RestrictionType.
        Raise a KeyError if other is not in B.
        """
        return set.remove(self, self.format(other))

    def add(self, other):
        """Add a restriction enzyme to the restriction batch.

        Safe set.add method. Verify that other is a RestrictionType or can be
        evaluated to a RestrictionType.
        Raise a ValueError if other can not be evaluated to a RestrictionType.
        """
        return set.add(self, self.format(other))

    def add_nocheck(self, other):
        """Add restriction enzyme to batch without checking its type."""
        return set.add(self, other)

    def format(self, y):
        """Evaluate enzyme (name) and return it (as RestrictionType).

        If y is a RestrictionType return y.
        If y can be evaluated to a RestrictionType return eval(y).
        Raise a ValueError in all other case.
        """
        try:
            if isinstance(y, RestrictionType):
                return y
            elif isinstance(eval(str(y)), RestrictionType):
                return eval(y)
        except (NameError, SyntaxError):
            pass
        raise ValueError(f'{y.__class__} is not a RestrictionType')

    def is_restriction(self, y):
        """Return if enzyme (name) is a known enzyme.

        True if y or eval(y) is a RestrictionType.
        """
        return isinstance(y, RestrictionType) or isinstance(eval(str(y)), RestrictionType)

    def split(self, *classes, **bool):
        """Extract enzymes of a certain class and put in new RestrictionBatch.

        It works but it is slow, so it has really an interest when splitting
        over multiple conditions.
        """

        def splittest(element):
            for klass in classes:
                b = bool.get(klass.__name__, True)
                if issubclass(element, klass):
                    if b:
                        continue
                    else:
                        return False
                elif b:
                    return False
                else:
                    continue
            return True
        d = list(filter(splittest, self))
        new = RestrictionBatch()
        new._data = dict(zip(d, [True] * len(d)))
        return new

    def elements(self):
        """List the enzymes of the RestrictionBatch as list of strings.

        Give all the names of the enzymes in B sorted alphabetically.
        """
        return sorted((str(e) for e in self))

    def as_string(self):
        """List the names of the enzymes of the RestrictionBatch.

        Return a list of the name of the elements of the batch.
        """
        return [str(e) for e in self]

    @classmethod
    def suppl_codes(cls):
        """Return a dictionary with supplier codes.

        Letter code for the suppliers.
        """
        supply = {k: v[0] for k, v in suppliers_dict.items()}
        return supply

    @classmethod
    def show_codes(cls):
        """Print a list of supplier codes."""
        supply = [' = '.join(i) for i in cls.suppl_codes().items()]
        print('\n'.join(supply))

    def search(self, dna, linear=True):
        """Return a dic of cutting sites in the seq for the batch enzymes."""
        if not hasattr(self, 'already_mapped'):
            self.already_mapped = None
        if isinstance(dna, DNA):
            if (str(dna), linear) == self.already_mapped:
                return self.mapping
            else:
                self.already_mapped = (str(dna), linear)
                fseq = FormattedSeq(dna, linear)
                self.mapping = {x: x.search(fseq) for x in self}
                return self.mapping
        elif isinstance(dna, FormattedSeq):
            if (str(dna), dna.linear) == self.already_mapped:
                return self.mapping
            else:
                self.already_mapped = (str(dna), dna.linear)
                self.mapping = {x: x.search(dna) for x in self}
                return self.mapping
        raise TypeError(f'Expected Seq or MutableSeq instance, got {type(dna)} instead')