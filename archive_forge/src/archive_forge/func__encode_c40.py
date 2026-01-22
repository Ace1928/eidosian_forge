from reportlab.graphics.barcode.common import Barcode
def _encode_c40(self, value):
    encoded = []
    for c in value:
        encoded += self._encode_c40_char(c)
    while len(encoded) % 3:
        encoded.append(0)
    codewords = []
    codewords.append(230)
    for i in range(0, len(encoded), 3):
        chunk = encoded[i:i + 3]
        total = chunk[0] * 1600 + chunk[1] * 40 + chunk[2] + 1
        codewords.append(total // 256)
        codewords.append(total % 256)
    codewords.append(254)
    if len(codewords) > self.cw_data:
        raise Exception('Too much data to fit into a data matrix of this size')
    if len(codewords) < self.cw_data:
        codewords.append(129)
        while len(codewords) < self.cw_data:
            r = 149 * (len(codewords) + 1) % 253 + 1
            codewords.append((129 + r) % 254)
    return codewords