import logging
def verboseLogger(self):
    """
        Print information about the iteration to the console
        """
    print('****** Iteration %d ******' % self.iteration)
    print('objectiveValue = %s' % self.objectiveValue)
    print('feasibility = %s' % self.feasibility)
    print('trustRadius = %s' % self.trustRadius)
    print('stepNorm = %s' % self.stepNorm)
    if self.fStep:
        print('INFO: f-type step')
    if self.thetaStep:
        print('INFO: theta-type step')
    if self.rejected:
        print('INFO: step rejected')
    print(25 * '*')