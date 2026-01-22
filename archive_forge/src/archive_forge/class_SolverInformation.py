import enum
from pyomo.opt.results.container import MapContainer, ScalarType
class SolverInformation(MapContainer):

    def __init__(self):
        MapContainer.__init__(self)
        self.declare('name')
        self.declare('status', value=SolverStatus.ok)
        self.declare('return_code')
        self.declare('message')
        self.declare('user_time', type=ScalarType.time)
        self.declare('system_time', type=ScalarType.time)
        self.declare('wallclock_time', type=ScalarType.time)
        self.declare('termination_condition', value=TerminationCondition.unknown)
        self.declare('termination_message')
        self.declare('statistics', value=SolverStatistics(), active=False)