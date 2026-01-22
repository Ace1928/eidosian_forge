class range_set(object):
    """A collection of range_specs, with units (e.g., bytes).
    """
    __slots__ = ['units', 'range_specs']

    def __init__(self):
        self.units = 'bytes'
        self.range_specs = []

    def __str__(self):
        return self.units + '=' + ', '.join([str(s) for s in self.range_specs])

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__, self.__class__.__name__, repr(self.__str__()))

    def from_str(self, s, valid_units=('bytes', 'none')):
        """Sets this range set based upon a string, such as the Range: header.

        You can also use the parse_range_set() function for more control.

        If a parsing error occurs, the pre-exising value of this range
        set is left unchanged.

        """
        r, k = parse_range_set(s, valid_units=valid_units)
        if k < len(s):
            raise ParseError('Extra unparsable characters in range set specifier', s, k)
        self.units = r.units
        self.range_specs = r.range_specs

    def is_single_range(self):
        """Does this range specifier consist of only a single range set?"""
        return len(self.range_specs) == 1

    def is_contiguous(self):
        """Can the collection of range_specs be coalesced into a single contiguous range?"""
        if len(self.range_specs) <= 1:
            return True
        merged = self.range_specs[0].copy()
        for s in self.range_specs[1:]:
            try:
                merged.merge_with(s)
            except:
                return False
        return True

    def fix_to_size(self, size):
        """Changes all length-relative range_specs to absolute range_specs based upon given file size.
        If none of the range_specs in this set can be satisfied, then the
        entire set is considered unsatifiable and an error is raised.
        Otherwise any unsatisfiable range_specs will simply be removed
        from this set.

        """
        for i in range(len(self.range_specs)):
            try:
                self.range_specs[i].fix_to_size(size)
            except RangeUnsatisfiableError:
                self.range_specs[i] = None
        self.range_specs = [s for s in self.range_specs if s is not None]
        if len(self.range_specs) == 0:
            raise RangeUnsatisfiableError('No ranges can be satisfied')

    def coalesce(self):
        """Collapses all consecutive range_specs which together define a contiguous range.

        Note though that this method will not re-sort the range_specs, so a
        potentially contiguous range may not be collapsed if they are
        not sorted.  For example the ranges:
            10-20, 30-40, 20-30
        will not be collapsed to just 10-40.  However if the ranges are
        sorted first as with:
            10-20, 20-30, 30-40
        then they will collapse to 10-40.
        """
        if len(self.range_specs) <= 1:
            return
        for i in range(len(self.range_specs) - 1):
            a = self.range_specs[i]
            b = self.range_specs[i + 1]
            if a is not None:
                try:
                    a.merge_with(b)
                    self.range_specs[i + 1] = None
                except RangeUnmergableError:
                    pass
        self.range_specs = [r for r in self.range_specs if r is not None]