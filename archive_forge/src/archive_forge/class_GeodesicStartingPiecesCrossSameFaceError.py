class GeodesicStartingPiecesCrossSameFaceError(DrillGeodesicError):

    def __init__(self):
        super().__init__('The first and last piece of the geodesic do not cross distinct faces. This can happen if the amount the start point was perturbed too much. Unfortunately, reducing this amount has not been implemented yet. If you run into this case, please report it giving the manifold and geodesic it occurred with.')