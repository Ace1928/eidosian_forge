from reportlab.graphics.barcode.common import Barcode
def _gfproduct(self, int1, int2):
    if int1 == 0 or int2 == 0:
        return 0
    else:
        return ALOGVAL[(LOGVAL[int1] + LOGVAL[int2]) % 255]