from reportlab.graphics.barcode.common import Barcode
def _encode_c40_char(self, char):
    o = ord(char)
    encoded = []
    if o == 32 or (o >= 48 and o <= 57) or (o >= 65 and o <= 90):
        if o == 32:
            encoded.append(o - 29)
        elif o >= 48 and o <= 57:
            encoded.append(o - 44)
        else:
            encoded.append(o - 51)
    elif o >= 0 and o <= 31:
        encoded.append(0)
        encoded.append(o)
    elif o >= 33 and o <= 64 or (o >= 91 and o <= 95):
        encoded.append(1)
        if o >= 33 and o <= 64:
            encoded.append(o - 33)
        else:
            encoded.append(o - 69)
    elif o >= 96 and o <= 127:
        encoded.append(2)
        encoded.append(o - 96)
    elif o >= 128 and o <= 255:
        encoded.append(1)
        encoded.append(30)
        encoded += self._encode_c40_char(chr(o - 128))
    else:
        raise Exception('Cannot encode %s (%s)' % (char, o))
    return encoded