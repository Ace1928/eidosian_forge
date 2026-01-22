import enum
from pyomo.opt.results.container import MapContainer, ScalarType
class BlackBoxStats(MapContainer):

    def __init__(self):
        MapContainer.__init__(self)
        self.declare('number of function evaluations')
        self.declare('number of gradient evaluations')
        self.declare('number of iterations')