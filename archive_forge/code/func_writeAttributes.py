def writeAttributes(self, attributes, data):
    if attributes is not None:
        for key, value in attributes.items():
            self.writeString(key, data)
            self.writeString(value, data, True)