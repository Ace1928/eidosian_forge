from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
class XmlView(Table):

    def __init__(self, source, *args, **kwargs):
        self.source = source
        self.args = args
        if len(args) == 2 and isinstance(args[1], (string_types, tuple, list)):
            self.rmatch = args[0]
            self.vmatch = args[1]
            self.vdict = None
            self.attr = None
        elif len(args) == 2 and isinstance(args[1], dict):
            self.rmatch = args[0]
            self.vmatch = None
            self.vdict = args[1]
            self.attr = None
        elif len(args) == 3:
            self.rmatch = args[0]
            self.vmatch = args[1]
            self.vdict = None
            self.attr = args[2]
        else:
            assert False, 'bad parameters'
        self.missing = kwargs.get('missing', None)
        self.user_parser = kwargs.get('parser', None)

    def __iter__(self):
        vmatch = self.vmatch
        vdict = self.vdict
        with self.source.open('rb') as xmlf:
            parser2 = _create_xml_parser(self.user_parser)
            tree = etree.parse(xmlf, parser=parser2)
            if not hasattr(tree, 'iterfind'):
                tree.iterfind = tree.findall
            if vmatch is not None:
                for rowelm in tree.iterfind(self.rmatch):
                    if self.attr is None:
                        getv = attrgetter('text')
                    else:
                        getv = lambda e: e.get(self.attr)
                    if isinstance(vmatch, string_types):
                        velms = rowelm.findall(vmatch)
                    else:
                        velms = itertools.chain(*[rowelm.findall(enm) for enm in vmatch])
                    yield tuple((getv(velm) for velm in velms))
            else:
                flds = tuple(sorted(map(text_type, vdict.keys())))
                yield flds
                vmatches = dict()
                vgetters = dict()
                for f in flds:
                    vmatch = self.vdict[f]
                    if isinstance(vmatch, string_types):
                        vmatches[f] = vmatch
                        vgetters[f] = element_text_getter(self.missing)
                    else:
                        vmatches[f] = vmatch[0]
                        attr = vmatch[1]
                        vgetters[f] = attribute_text_getter(attr, self.missing)
                for rowelm in tree.iterfind(self.rmatch):
                    yield tuple((vgetters[f](rowelm.findall(vmatches[f])) for f in flds))