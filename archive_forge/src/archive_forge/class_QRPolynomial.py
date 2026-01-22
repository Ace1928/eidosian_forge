import re
import itertools
class QRPolynomial:

    def __init__(self, num, shift):
        if len(num) == 0:
            raise Exception(len(num) + '/' + shift)
        offset = 0
        while offset < len(num) and num[offset] == 0:
            offset += 1
        self.num = num[offset:] + [0] * shift

    def get(self, index):
        return self.num[index]

    def getLength(self):
        return len(self.num)

    def multiply(self, e):
        num = [0] * (self.getLength() + e.getLength() - 1)
        for i in range(self.getLength()):
            for j in range(e.getLength()):
                num[i + j] ^= QRMath.gexp(QRMath.glog(self.get(i)) + QRMath.glog(e.get(j)))
        return QRPolynomial(num, 0)

    def mod(self, e):
        if self.getLength() < e.getLength():
            return self
        ratio = QRMath.glog(self.num[0]) - QRMath.glog(e.num[0])
        num = [nn ^ QRMath.gexp(QRMath.glog(en) + ratio) for nn, en in zip(self.num, e.num)]
        num += self.num[e.getLength():]
        return QRPolynomial(num, 0).mod(e)