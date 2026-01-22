class ReplacedBy:
    prefix = '- '
    suffix = ''

    def __init__(self, chunk, total_count):
        self.chunk = chunk
        self.total_count = total_count

    def __iter__(self):
        lines = [self.prefix + str(item) + self.suffix for item in self.chunk]
        while len(lines) < self.total_count:
            lines.append('')
        return iter(lines)