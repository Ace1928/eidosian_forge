class ds:
    """DS lines, include file name of a read's chromatogram file."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.chromat_file = ''
        self.phd_file = ''
        self.time = ''
        self.chem = ''
        self.dye = ''
        self.template = ''
        self.direction = ''
        if line:
            tags = ['CHROMAT_FILE', 'PHD_FILE', 'TIME', 'CHEM', 'DYE', 'TEMPLATE', 'DIRECTION']
            poss = [line.find(x) for x in tags]
            tagpos = dict(zip(poss, tags))
            if -1 in tagpos:
                del tagpos[-1]
            ps = sorted(tagpos)
            for p1, p2 in zip(ps, ps[1:] + [len(line) + 1]):
                setattr(self, tagpos[p1].lower(), line[p1 + len(tagpos[p1]) + 1:p2].strip())