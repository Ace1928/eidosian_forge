def tryPackAndWriteHeader(self, v, headerData, data):
    size = len(headerData)
    if size >= 128:
        return None
    arr = [0] * int((size + 1) / 2)
    for i in range(0, size):
        packByte = self.packByte(v, headerData[i])
        if packByte == -1:
            arr = []
            break
        n2 = int(i / 2)
        arr[n2] |= packByte << 4 * (1 - i % 2)
    if len(arr) > 0:
        if size % 2 == 1:
            arr[-1] |= 15
        data.append(v)
        self.writeInt8(size % 2 << 7 | len(arr), data)
        return arr
    return None