class GeodesicSystemNotSimpleError(DrillGeodesicError):

    def __init__(self, maximal_tube_radius):
        self.maximal_tube_radius = maximal_tube_radius
        super().__init__('One of the given geodesics might not simple or two of the given geodesics might intersect. The maximal tube radius about the given system of geodesics was estimated to be: %r.' % maximal_tube_radius)