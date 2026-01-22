class RunIterator(AbstractRunIterator):

    def __init__(self, run_list):
        self._run_list_iter = iter(run_list)
        self.start, self.end, self.value = next(self)

    def __next__(self):
        return next(self._run_list_iter)

    def __getitem__(self, index):
        try:
            while index >= self.end and index > self.start:
                self.start, self.end, self.value = next(self)
            return self.value
        except StopIteration:
            raise IndexError

    def ranges(self, start, end):
        try:
            while start >= self.end:
                self.start, self.end, self.value = next(self)
            yield (start, min(self.end, end), self.value)
            while end > self.end:
                self.start, self.end, self.value = next(self)
                yield (self.start, min(self.end, end), self.value)
        except StopIteration:
            return