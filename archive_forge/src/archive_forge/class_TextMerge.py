class TextMerge:
    """Base class for text-mergers
    Subclasses must implement _merge_struct.

    Many methods produce or consume structured merge information.
    This is an iterable of tuples of lists of lines.
    Each tuple may have a length of 1 - 3, depending on whether the region it
    represents is conflicted.

    Unconflicted region tuples have length 1.
    Conflicted region tuples have length 2 or 3.  Index 1 is text_a, e.g. THIS.
    Index 1 is text_b, e.g. OTHER.  Index 2 is optional.  If present, it
    represents BASE.
    """
    A_MARKER = b'<<<<<<< \n'
    B_MARKER = b'>>>>>>> \n'
    SPLIT_MARKER = b'=======\n'

    def __init__(self, a_marker=A_MARKER, b_marker=B_MARKER, split_marker=SPLIT_MARKER):
        self.a_marker = a_marker
        self.b_marker = b_marker
        self.split_marker = split_marker

    def _merge_struct(self):
        """Return structured merge info.  Must be implemented by subclasses.
        See TextMerge docstring for details on the format.
        """
        raise NotImplementedError('_merge_struct is abstract')

    def struct_to_lines(self, struct_iter):
        """Convert merge result tuples to lines"""
        for lines in struct_iter:
            if len(lines) == 1:
                yield from lines[0]
            else:
                yield self.a_marker
                yield from lines[0]
                yield self.split_marker
                yield from lines[1]
                yield self.b_marker

    def iter_useful(self, struct_iter):
        """Iterate through input tuples, skipping empty ones."""
        for group in struct_iter:
            if len(group[0]) > 0:
                yield group
            elif len(group) > 1 and len(group[1]) > 0:
                yield group

    def merge_lines(self, reprocess=False):
        """Produce an iterable of lines, suitable for writing to a file
        Returns a tuple of (line iterable, conflict indicator)
        If reprocess is True, a two-way merge will be performed on the
        intermediate structure, to reduce conflict regions.
        """
        struct = []
        conflicts = False
        for group in self.merge_struct(reprocess):
            struct.append(group)
            if len(group) > 1:
                conflicts = True
        return (self.struct_to_lines(struct), conflicts)

    def merge_struct(self, reprocess=False):
        """Produce structured merge info"""
        struct_iter = self.iter_useful(self._merge_struct())
        if reprocess is True:
            return self.reprocess_struct(struct_iter)
        else:
            return struct_iter

    @staticmethod
    def reprocess_struct(struct_iter):
        """ Perform a two-way merge on structural merge info.
        This reduces the size of conflict regions, but breaks the connection
        between the BASE text and the conflict region.

        This process may split a single conflict region into several smaller
        ones, but will not introduce new conflicts.
        """
        for group in struct_iter:
            if len(group) == 1:
                yield group
            else:
                yield from Merge2(group[0], group[1]).merge_struct()