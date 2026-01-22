def all_superiors(self, include_wildcard=False):
    """Returns a list of this language and all it's superiors.

        If include_wildcard is False, then "*" will not be among the
        output list, unless this language is itself "*".

        """
    langlist = [self]
    l = self
    while not l.is_universal_wildcard():
        l = l.superior()
        if l.is_universal_wildcard() and (not include_wildcard):
            continue
        langlist.append(l)
    return langlist