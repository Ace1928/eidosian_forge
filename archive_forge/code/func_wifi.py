from .command import Command
@property
def wifi(self):
    return self.mask / 2 % 2 == 1