import enum
from pyomo.opt.results.container import MapContainer, ScalarType
class BranchAndBoundStats(MapContainer):

    def __init__(self):
        MapContainer.__init__(self)
        self.declare('number of bounded subproblems')
        self.declare('number of created subproblems')