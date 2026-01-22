class CharInfoWord(object):

    def __init__(self, word):
        b1, b2, b3, b4 = (word >> 24, (word & 16711680) >> 16, (word & 65280) >> 8, word & 255)
        self.width_index = b1
        self.height_index = b2 >> 4
        self.depth_index = b2 & 15
        self.italic_index = (b3 & 252) >> 2
        self.tag = b3 & 3
        self.remainder = b4

    def has_ligkern(self):
        return self.tag == 1

    def ligkern_start(self):
        return self.remainder