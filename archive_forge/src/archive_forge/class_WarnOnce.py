from sys import stderr
class WarnOnce:

    def __init__(self, kind='Warn'):
        self.uttered = {}
        self.pfx = '%s: ' % kind
        self.enabled = 1

    def once(self, warning):
        if warning not in self.uttered:
            if self.enabled:
                logger.write(self.pfx + warning)
            self.uttered[warning] = 1

    def __call__(self, warning):
        self.once(warning)