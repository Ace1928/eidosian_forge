def _index_diff(self, x, y=0.0):
    return int((x - y + 0.5 * self.step) // self.step)