from rdkit.sping.colors import *
class AffineMatrix:

    def __init__(self, init=None):
        if init:
            if len(init) == 6:
                self.A = init
            if type(init) == type(self):
                self.A = init.A
        else:
            self.A = [1.0, 0, 0, 1.0, 0.0, 0.0]

    def scale(self, sx, sy):
        self.A = [sx * self.A[0], sx * self.A[1], sy * self.A[2], sy * self.A[3], self.A[4], self.A[5]]

    def rotate(self, theta):
        """counter clockwise rotation in standard SVG/libart coordinate system"""
        co = math.cos(PI * theta / 180.0)
        si = math.sin(PI * theta / 180.0)
        self.A = [self.A[0] * co + self.A[2] * si, self.A[1] * co + self.A[3] * si, -self.A[0] * si + self.A[2] * co, -self.A[1] * si + self.A[3] * co, self.A[4], self.A[5]]

    def translate(self, tx, ty):
        self.A = [self.A[0], self.A[1], self.A[2], self.A[3], self.A[0] * tx + self.A[2] * ty + self.A[4], self.A[1] * tx + self.A[3] * ty + self.A[5]]