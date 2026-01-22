import random
def AddText(self, x, y, label):
    text = self.__dwg.text(label, insert=(x * self.__scaling + self.__offset, (self.__sizey - y) * self.__scaling + self.__offset), text_anchor='middle', font_family='sans-serif', font_size='%dpx' % (self.__scaling / 2))
    self.__dwg.add(text)