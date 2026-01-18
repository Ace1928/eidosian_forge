def writeInt24(self, v, data):
    data.append((v & 16711680) >> 16)
    data.append((v & 65280) >> 8)
    data.append((v & 255) >> 0)