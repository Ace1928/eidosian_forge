from prov.model import *
def test_usage_3(self):
    document = self.new_document()
    use = document.usage(EX_NS['a1'], entity=EX_NS['e1'], identifier=EX_NS['use3'])
    use.add_attributes([(PROV_ROLE, 'somerole'), (PROV_ROLE, 'otherRole')])
    self.do_tests(document)