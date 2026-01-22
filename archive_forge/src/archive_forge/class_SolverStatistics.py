import enum
from pyomo.opt.results.container import MapContainer, ScalarType
class SolverStatistics(MapContainer):

    def __init__(self):
        MapContainer.__init__(self)
        self.declare('branch_and_bound', value=BranchAndBoundStats(), active=False)
        self.declare('black_box', value=BlackBoxStats(), active=False)