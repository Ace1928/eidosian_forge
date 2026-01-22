from __future__ import absolute_import, print_function, division
import csv
import cStringIO
from petl.util.base import Table, data
from petl.io.base import getcodec
class UnicodeWriter:

    def __init__(self, buf, encoding=None, errors='strict', **csvargs):
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, **csvargs)
        self.stream = buf
        codec = getcodec(encoding)
        self.encoder = codec.incrementalencoder(errors)

    def writerow(self, row):
        self.writer.writerow([unicode(s).encode('utf-8') if s is not None else None for s in row])
        data = self.queue.getvalue()
        data = data.decode('utf-8')
        data = self.encoder.encode(data)
        self.stream.write(data)
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)