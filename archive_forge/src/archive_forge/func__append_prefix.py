from enum import Enum
def _append_prefix(self, prefix):
    """Append PREFIX."""
    if len(prefix) > 0:
        self.args.append('PREFIX')
        self.args.append(len(prefix))
        for p in prefix:
            self.args.append(p)