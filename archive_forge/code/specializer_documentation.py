from fontTools.cffLib import maxStackLimit

    Takes X,Y vector v and returns one of r, h, v, or 0 depending on which
    of X and/or Y are zero, plus tuple of nonzero ones.  If both are zero,
    it returns a single zero still.

    >>> _categorizeVector((0,0))
    ('0', (0,))
    >>> _categorizeVector((1,0))
    ('h', (1,))
    >>> _categorizeVector((0,2))
    ('v', (2,))
    >>> _categorizeVector((1,2))
    ('r', (1, 2))
    