from heapq import heappop, heappush
from shapely.errors import TopologicalError
from shapely.geometry import Point
Signed distance from Cell centroid to polygon outline. The returned
        value is negative if the point is outside of the polygon exterior
        boundary.
        