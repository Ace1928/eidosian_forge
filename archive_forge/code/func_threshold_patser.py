def threshold_patser(self):
    """Threshold selection mimicking the behaviour of patser (Hertz, Stormo 1999) software.

        It selects such a threshold that the log(fpr)=-ic(M)
        note: the actual patser software uses natural logarithms instead of log_2, so the numbers
        are not directly comparable.
        """
    return self.threshold_fpr(fpr=2 ** (-self.ic))