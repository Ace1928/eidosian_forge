import re
def _make_map_only(self, ls, title, nc=(), s1=''):
    """Make string describing cutting map (PRIVATE).

        Return a string of form::

            | title.
            |
            |     enzyme1, position
            |     |
            | AAAAAAAAAAAAAAAAAAAAA...
            | |||||||||||||||||||||
            | TTTTTTTTTTTTTTTTTTTTT...

        Arguments:
         - ls is a list of results.
         - title is a string.
         - Non cutting enzymes are not included.
        """
    if not ls:
        return title
    resultKeys = sorted((str(x) for x, y in ls))
    map = title or ''
    enzymemap = {}
    for enzyme, cut in ls:
        for c in cut:
            if c in enzymemap:
                enzymemap[c].append(str(enzyme))
            else:
                enzymemap[c] = [str(enzyme)]
    mapping = sorted(enzymemap.keys())
    cutloc = {}
    x, counter, length = (0, 0, len(self.sequence))
    for x in range(60, length, 60):
        counter = x - 60
        loc = []
        cutloc[counter] = loc
        remaining = []
        for key in mapping:
            if key <= x:
                loc.append(key)
            else:
                remaining.append(key)
        mapping = remaining
    cutloc[x] = mapping
    sequence = str(self.sequence)
    revsequence = str(self.sequence.complement())
    a = '|'
    base, counter = (0, 0)
    emptyline = ' ' * 60
    Join = ''.join
    for base in range(60, length, 60):
        counter = base - 60
        line = emptyline
        for key in cutloc[counter]:
            s = ''
            if key == base:
                for n in enzymemap[key]:
                    s = ' '.join((s, n))
                chunk = line[0:59]
                lineo = Join((chunk, str(key), s, '\n'))
                line2 = Join((chunk, a, '\n'))
                linetot = Join((lineo, line2))
                map = Join((map, linetot))
                break
            for n in enzymemap[key]:
                s = ' '.join((s, n))
            k = key % 60
            lineo = Join((line[0:k - 1], str(key), s, '\n'))
            line = Join((line[0:k - 1], a, line[k:]))
            line2 = Join((line[0:k - 1], a, line[k:], '\n'))
            linetot = Join((lineo, line2))
            map = Join((map, linetot))
        mapunit = '\n'.join((sequence[counter:base], a * 60, revsequence[counter:base], Join((str.ljust(str(counter + 1), 15), ' ' * 30, str.rjust(str(base), 15), '\n\n'))))
        map = Join((map, mapunit))
    line = ' ' * 60
    for key in cutloc[base]:
        s = ''
        if key == length:
            for n in enzymemap[key]:
                s = Join((s, ' ', n))
            chunk = line[0:length - 1]
            lineo = Join((chunk, str(key), s, '\n'))
            line2 = Join((chunk, a, '\n'))
            linetot = Join((lineo, line2))
            map = Join((map, linetot))
            break
        for n in enzymemap[key]:
            s = Join((s, ' ', n))
        k = key % 60
        lineo = Join((line[0:k - 1], str(key), s, '\n'))
        line = Join((line[0:k - 1], a, line[k:]))
        line2 = Join((line[0:k - 1], a, line[k:], '\n'))
        linetot = Join((lineo, line2))
        map = Join((map, linetot))
    mapunit = ''
    mapunit = Join((sequence[base:length], '\n'))
    mapunit = Join((mapunit, a * (length - base), '\n'))
    mapunit = Join((mapunit, revsequence[base:length], '\n'))
    mapunit = Join((mapunit, Join((str.ljust(str(base + 1), 15), ' ' * (length - base - 30), str.rjust(str(length), 15), '\n\n'))))
    map = Join((map, mapunit))
    return map