class ECrossing:
    """
    A pair: (Crossing, Arrow), where the Arrow is involved in the Crossing.
    The ECrossings correspond 1-1 with edges of the link diagram.
    """

    def __init__(self, crossing, arrow):
        if arrow not in crossing:
            raise ValueError
        self.crossing = crossing
        self.arrow = arrow
        self.strand = self.crossing.strand(self.arrow)

    def pair(self):
        return (self.crossing, self.arrow)

    def goes_over(self):
        return self.arrow == self.crossing.over