from prov.model import *
def test_start_1(self):
    document = self.new_document()
    document.start(None, trigger=EX_NS['e1'], identifier=EX_NS['start1'])
    self.do_tests(document)