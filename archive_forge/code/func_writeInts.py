import numpy
def writeInts(self, ints, prec='i'):
    """Write an array of integers in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec : string
            Character code for the precision to use in writing
        """
    if prec not in self._int_precisions:
        raise ValueError('Not an appropriate precision')
    nums = numpy.array(ints, dtype=self.ENDIAN + prec)
    self.writeRecord(nums.tostring())