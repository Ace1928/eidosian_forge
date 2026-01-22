class ShapeWithSkeleton(object):
    grid = None
    skelPts = None

    def __init__(self, *args, **kwargs):
        self._initMemberData()

    def _initMemberData(self):
        self.skelPts = []