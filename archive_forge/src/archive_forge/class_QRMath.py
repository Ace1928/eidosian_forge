import re
import itertools
class QRMath:

    @staticmethod
    def glog(n):
        if n < 1:
            raise Exception('glog(' + n + ')')
        return LOG_TABLE[n]

    @staticmethod
    def gexp(n):
        while n < 0:
            n += 255
        while n >= 256:
            n -= 255
        return EXP_TABLE[n]