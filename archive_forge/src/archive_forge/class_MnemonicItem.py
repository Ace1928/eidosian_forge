from collections import namedtuple
class MnemonicItem(object):
    """ Single mnemonic item.
    """

    def __init__(self, bytecode, mnemonic):
        self.bytecode = bytecode
        self.mnemonic = mnemonic

    def __repr__(self):
        return '%s ; %s' % (' '.join(['0x%02x' % x for x in self.bytecode]), self.mnemonic)