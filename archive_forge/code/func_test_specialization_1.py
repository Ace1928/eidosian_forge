from prov.model import *
def test_specialization_1(self):
    document = self.new_document()
    document.specialization(EX_NS['e2'], EX_NS['e1'])
    self.do_tests(document)