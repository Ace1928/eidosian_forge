class UnfinishedTraceGeodesicError(DrillGeodesicError):

    def __init__(self, steps):
        self.steps = steps
        super().__init__('The geodesic seems to have more than %d pieces in the triangulation. This is probably due to a pathology, e.g., the geodesic is very close to a core curve of filled cusp.' % steps)