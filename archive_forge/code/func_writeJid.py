def writeJid(self, user, server, data):
    data.append(250)
    if user is not None:
        self.writeString(user, data, True)
    else:
        self.writeToken(0, data)
    self.writeString(server, data)