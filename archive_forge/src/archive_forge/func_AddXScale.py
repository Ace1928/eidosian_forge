import random
def AddXScale(self, step=1):
    """Add an scale on the x axis."""
    o = self.__offset
    s = self.__scaling
    y = self.__sizey * s + o / 2.0 + o
    dy = self.__offset / 4.0
    self.__dwg.add(self.__dwg.line((o, y), (self.__sizex * s + o, y), stroke='black'))
    for i in range(0, int(self.__sizex) + 1, step):
        self.__dwg.add(self.__dwg.line((o + i * s, y - dy), (o + i * s, y + dy), stroke='black'))