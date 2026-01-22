class EventSet(list):

    def startElement(self, name, attrs, connection):
        if name == 'item':
            event = Event()
            self.append(event)
            return event
        else:
            return None

    def endElement(self, name, value, connection):
        setattr(self, name, value)