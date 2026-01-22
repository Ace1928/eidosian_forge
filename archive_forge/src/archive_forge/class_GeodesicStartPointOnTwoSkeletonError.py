class GeodesicStartPointOnTwoSkeletonError(GeodesicHittingOneSkeletonError):
    """
    Raised when the start point given to GeodesicInfo appears not to be in the
    interior of a tetrahedron.
    """