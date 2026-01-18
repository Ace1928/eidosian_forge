import operator
import math
def xyz_to_luv(triple):
    X, Y, Z = triple
    if X == Y == Z == 0.0:
        return [0.0, 0.0, 0.0]
    varU = 4.0 * X / (X + 15.0 * Y + 3.0 * Z)
    varV = 9.0 * Y / (X + 15.0 * Y + 3.0 * Z)
    L = 116.0 * f(Y / refY) - 16.0
    if L == 0.0:
        return [0.0, 0.0, 0.0]
    U = 13.0 * L * (varU - refU)
    V = 13.0 * L * (varV - refV)
    return [L, U, V]