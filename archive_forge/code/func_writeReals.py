import numpy
def writeReals(self, reals, prec='f'):
    """Write an array of floats in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec` : string
            Character code for the precision to use in writing
        """
    if prec not in self._real_precisions:
        raise ValueError('Not an appropriate precision')
    nums = numpy.array(reals, dtype=self.ENDIAN + prec)
    self.writeRecord(nums.tostring())