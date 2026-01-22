from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddFilter(args, noun):
    predicate = 'metadata.target ~ projects/\\d+/locations/.+/{}*'.format(noun)
    if args.filter:
        args.filter = predicate + ' AND ' + args.filter
    else:
        args.filter = predicate