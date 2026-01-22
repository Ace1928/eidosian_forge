from .meta import MetaMessage
class MidiTrack(list):

    @property
    def name(self):
        """Name of the track.

        This will return the name from the first track_name meta
        message in the track, or '' if there is no such message.

        Setting this property will update the name field of the first
        track_name message in the track. If no such message is found,
        one will be added to the beginning of the track with a delta
        time of 0."""
        for message in self:
            if message.type == 'track_name':
                return message.name
        else:
            return ''

    @name.setter
    def name(self, name):
        for message in self:
            if message.type == 'track_name':
                message.name = name
                return
        else:
            self.insert(0, MetaMessage('track_name', name=name, time=0))

    def copy(self):
        return self.__class__(self)

    def __getitem__(self, index_or_slice):
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            return lst
        else:
            return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __repr__(self):
        if len(self) == 0:
            messages = ''
        elif len(self) == 1:
            messages = f'[{self[0]}]'
        else:
            messages = '[\n  {}]'.format(',\n  '.join((repr(m) for m in self)))
        return f'{self.__class__.__name__}({messages})'