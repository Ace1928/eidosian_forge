import sys
from io import StringIO
def _repr_mime_(self, mime):
    if mime not in self.data:
        return
    data = self.data[mime]
    if mime in self.metadata:
        return (data, self.metadata[mime])
    else:
        return data