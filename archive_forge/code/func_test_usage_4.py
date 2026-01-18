from prov.model import *
def test_usage_4(self):
    document = self.new_document()
    use = document.usage(EX_NS['a1'], entity=EX_NS['e1'], identifier=EX_NS['use4'], time=datetime.datetime.now())
    use.add_attributes([(PROV_ROLE, 'somerole')])
    self.do_tests(document)