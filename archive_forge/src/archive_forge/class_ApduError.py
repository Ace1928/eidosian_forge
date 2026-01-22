class ApduError(HardwareError):

    def __init__(self, sw1, sw2):
        self.sw1 = sw1
        self.sw2 = sw2
        super(ApduError, self).__init__('Device returned status: %d %d' % (sw1, sw2))