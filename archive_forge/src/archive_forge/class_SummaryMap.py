class SummaryMap(dict):

    def __init__(self, parent=None):
        self.parent = parent
        dict.__init__(self)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'key':
            self._name = value
        elif name == 'value':
            try:
                self[self._name] = int(value)
            except ValueError:
                self[self._name] = value
        else:
            setattr(self, name, value)