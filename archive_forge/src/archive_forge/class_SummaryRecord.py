import typing as t
class SummaryRecord:
    """Encodes a diff -- analogous to the SummaryRecord protobuf message."""
    update: t.List['SummaryItem']
    remove: t.List['SummaryItem']

    def __init__(self):
        self.update = []
        self.remove = []

    def __str__(self):
        s = 'SummaryRecord:\n  Update:\n    '
        s += '\n    '.join([str(item) for item in self.update])
        s += '\n  Remove:\n    '
        s += '\n    '.join([str(item) for item in self.remove])
        s += '\n'
        return s
    __repr__ = __str__

    def _add_next_parent(self, parent_key):
        with_next_parent = SummaryRecord()
        with_next_parent.update = [item._add_next_parent(parent_key) for item in self.update]
        with_next_parent.remove = [item._add_next_parent(parent_key) for item in self.remove]
        return with_next_parent