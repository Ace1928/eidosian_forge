def writeListStart(self, i, data):
    if i == 0:
        data.append(0)
    elif i < 256:
        data.append(248)
        self.writeInt8(i, data)
    else:
        data.append(249)
        self.writeInt16(i, data)