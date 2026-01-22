from .command import Command
class _ConnectionType:

    def __init__(self, mask):
        self.mask = mask

    @property
    def airplane_mode(self):
        return self.mask % 2 == 1

    @property
    def wifi(self):
        return self.mask / 2 % 2 == 1

    @property
    def data(self):
        return self.mask / 4 > 0