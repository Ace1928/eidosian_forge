from prov.model import *
def test_start_4(self):
    document = self.new_document()
    document.start(None, trigger=EX_NS['e1'], identifier=EX_NS['start4'], starter=EX_NS['a2'])
    self.do_tests(document)