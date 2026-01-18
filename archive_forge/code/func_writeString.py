def writeString(self, tag, data, packed=False):
    tok = self.tokenDictionary.getIndex(tag)
    if tok:
        index, secondary = tok
        if not secondary:
            self.writeToken(index, data)
        else:
            quotient = index // 256
            if quotient == 0:
                double_byte_token = 236
            elif quotient == 1:
                double_byte_token = 237
            elif quotient == 2:
                double_byte_token = 238
            elif quotient == 3:
                double_byte_token = 239
            else:
                raise ValueError('Double byte dictionary token out of range')
            self.writeToken(double_byte_token, data)
            self.writeToken(index % 256, data)
    else:
        at = '@'.encode() if type(tag) == bytes else '@'
        try:
            atIndex = tag.index(at)
            if atIndex < 1:
                raise ValueError('atIndex < 1')
            else:
                server = tag[atIndex + 1:]
                user = tag[0:atIndex]
                self.writeJid(user, server, data)
        except ValueError:
            self.writeBytes(self.encodeString(tag), data, packed)