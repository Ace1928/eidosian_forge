from heapq import heappop, heappush
from shapely.errors import TopologicalError
from shapely.geometry import Point
def _dist(self, polygon):
    """Signed distance from Cell centroid to polygon outline. The returned
        value is negative if the point is outside of the polygon exterior
        boundary.
        """
    inside = polygon.contains(self.centroid)
    distance = self.centroid.distance(polygon.exterior)
    for interior in polygon.interiors:
        distance = min(distance, self.centroid.distance(interior))
    if inside:
        return distance
    return -distance