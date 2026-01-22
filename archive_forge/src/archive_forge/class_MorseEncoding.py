from .links_base import Strand, Crossing, Link
import random
import collections
class MorseEncoding:
    """
    A MorseEncoding is a concrete encoding of a Morse diagram of an
    oriented link in the sprit of:

    http://katlas.org/wiki/MorseLink_Presentations

    as a sequence of events.  At each stage, there are n active strand
    ends labeled in order [0, 1, ... , n - 1].  The events can be
    specified either as instances of Event or as tuples as shown
    below.

    >>> events = 2*[('cup', 0, 1)] + 2*[('cross', 1, 2)] + 2*[('cross', 3, 2)]
    >>> events += [('cap', 1, 2), ('cap', 0, 1)]
    >>> me = MorseEncoding(events)
    >>> me.events[1:3]
    [('cup', 0, 1), ('cross', 1, 2)]
    >>> me.width
    4
    >>> me_copy = MorseEncoding(me.events)
    >>> L = me.link()
    >>> L.exterior().identify()[0]  #doctest: +SNAPPY
    m004(0,0)
    >>> U = MorseEncoding([('cup', 0, 1), ('cap', 0, 1)]).link()
    >>> U.exterior().fundamental_group().num_generators()  #doctest: +SNAPPY
    1
    """

    def __init__(self, events):
        self.events = []
        for event in events:
            if not isinstance(event, Event):
                event = Event(*event)
            self.events.append(event)
        self._check_and_set_width()

    def _check_and_set_width(self):
        width = 0
        max_width = 0
        for event in self.events:
            if event.kind == 'cup':
                assert event.max < width + 2
                width += 2
                max_width = max(width, max_width)
            elif event.kind == 'cap':
                assert event.max < width
                width += -2
            elif event.kind == 'cross':
                assert event.max < width
        assert width == 0
        self.width = max_width

    def link(self):
        active = {}
        crossings = []
        for event in self.events:
            if event.kind == 'cup':
                S = Strand()
                crossings.append(S)
                active = insert_space(active, event.min)
                active[event.a] = S[0]
                active[event.b] = S[1]
            elif event.kind == 'cap':
                S = Strand()
                crossings.append(S)
                S[0] = active[event.a]
                S[1] = active[event.b]
                active = remove_space(active, event.min)
            elif event.kind == 'cross':
                C = Crossing()
                crossings.append(C)
                if event.a < event.b:
                    C[3] = active[event.a]
                    C[0] = active[event.b]
                    active[event.a] = C[2]
                    active[event.b] = C[1]
                else:
                    C[3] = active[event.a]
                    C[2] = active[event.b]
                    active[event.a] = C[0]
                    active[event.b] = C[1]
        return Link(crossings)

    def __iter__(self):
        return self.events.__iter__()