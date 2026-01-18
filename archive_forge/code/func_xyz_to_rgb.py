import operator
import math
def xyz_to_rgb(triple):
    xyz = map(lambda row: dot_product(row, triple), m)
    return list(map(from_linear, xyz))