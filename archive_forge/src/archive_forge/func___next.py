from ._constants import *
def __next(self):
    index = self.index
    try:
        char = self.decoded_string[index]
    except IndexError:
        self.next = None
        return
    if char == '\\':
        index += 1
        try:
            char += self.decoded_string[index]
        except IndexError:
            raise error('bad escape (end of pattern)', self.string, len(self.string) - 1) from None
    self.index = index + 1
    self.next = char