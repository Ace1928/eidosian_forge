from fontTools.misc.textTools import bytechr, bytesjoin, byteord
def _decryptChar(cipher, R):
    cipher = byteord(cipher)
    plain = (cipher ^ R >> 8) & 255
    R = (cipher + R) * 52845 + 22719 & 65535
    return (bytechr(plain), R)