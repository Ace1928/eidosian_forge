import logging
def updateIteration(self, feasibility=None, objectiveValue=None, trustRadius=None, stepNorm=None):
    """
        Update values in current record
        """
    if feasibility is not None:
        self.iterrecord.feasibility = feasibility
    if objectiveValue is not None:
        self.iterrecord.objectiveValue = objectiveValue
    if trustRadius is not None:
        self.iterrecord.trustRadius = trustRadius
    if stepNorm is not None:
        self.iterrecord.stepNorm = stepNorm