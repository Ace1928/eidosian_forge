from prov.model import *
def test_usage_2(self):
    document = self.new_document()
    document.usage(EX_NS['a1'], entity=EX_NS['e1'], identifier=EX_NS['use2'])
    self.do_tests(document)